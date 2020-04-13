import math
from functools import partial

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from ui import graph_manager as manager
from ui import specific_widgets
from ui.widgets.custom_widgets import create_rv_data
from ui.widgets.custom_widgets import NodeColor
from ui.widgets.custom_widgets import NodeDisplay
from ui.widgets.custom_widgets import NodeMultiWidget
from ui.widgets.custom_widgets import NodeRV
from ui.widgets.custom_widgets import NodeSlider
from ui.widgets.custom_widgets import NodeSwitch


class NodePage(manager.InputInterface, FloatLayout):
    def __init__(self, node, node_name, graph_manager, **kwargs):
        """
        Create a page for a specific graph node. Thereby add a widget for each parameter according to the value type:
        Currently supported types:

            int: Slider
            base2Int: Slider
            float: Slider
            bool: Switch
            str: TextInput
            color: ColorWheel
            list: Recycle view

        Changes made to any node have to be communicated via the graph manager.
        Incoming changes will be handled in the update_observable function.

        Args:
            node: Current node.
            node_name: Name of the node.
            graph_manager: Graph manager this page is subscribing to.
            **kwargs:
        """
        super(NodePage, self).__init__(graph_manager)
        self.node = node
        self.node_name = node_name
        self.node_behaviours = self.graph_manager.node_regulations

        # store all widgets of the page, to be able to update values.
        self.widget_dic = {}
        for attributes in self.node.values():
            name = attributes.name
            value = attributes.value
            type_specs = self.node_behaviours[self.node_name].inputs[name]

            if name in specific_widgets:
                widget = self.generate_specific_widget(
                    type_specs=type_specs,
                    value=value,
                    name=name,
                    attributes=attributes,
                )
            else:
                widget = self.generate_generic_widget(
                    type_specs=type_specs, value=value, name=name, set_value=self.set
                )
                attributes.subscribe(self.update_observable)
            self.create_widget(name, widget)

    def generate_specific_widget(
        self, type_specs=None, value=None, name: str = "", attributes=None,
    ) -> Widget:
        """
        Create a widget that is more complex then the general widgets and has its own set up.

        Args:
            type_specs: Specified type.
            value: Value.
            name: Parameter name.
            attributes: Observable node.

        Returns:
            A specific Widget.

        """
        widget_type = specific_widgets[name]
        widget = widget_type(
            node_name=self.node_name,
            param_name=name,
            value=value,
            graph_manager=self.graph_manager,
            attributes=attributes,
        )
        return widget

    def generate_generic_widget(
        self, type_specs=None, value=None, name: str = "", set_value=None
    ) -> Widget:
        """
        Generate a generic widget depending on the type, with the possible for minor specifications.

        Args:
            type_specs: Specifications for the type currently including min, max.
            value: Value of the parameter.
            name: Name of the parameter that is adjusted.

        Returns:
            Widget of a generic style (Slider, Switch, Textinut ...).
        """
        if type_specs is None:
            type_specs = {}
        widget_type = None
        specific_params = {}
        typename = type_specs["typename"]
        if typename == "Bool":
            widget_type = NodeSwitch
        elif typename == "UInt":
            set_value = partial(self.modify_set, type=int)
            min_value = type_specs.get("lower_bound", 0)
            max_value = type_specs.get("upper_bound", 200)
            specific_params = {"min_value": min_value, "max_value": max_value}
            widget_type = NodeSlider
        elif typename == "Float":
            min_value = type_specs.get("lower_bound", 0)
            max_value = type_specs.get("upper_bound", 200)
            step = 0.1
            specific_params = {
                "min_value": min_value,
                "max_value": max_value,
                "step": step,
            }
            widget_type = NodeSlider
        elif typename == "Base2Int":
            set_value = partial(self.modify_set, value_type=int)
            min_value = math.log(type_specs.get("lower_bound", 1), 2)
            max_value = math.log(type_specs.get("upper_bound", 128), 2)
            specific_params = {
                "min_value": min_value,
                "max_value": max_value,
                "slider_transform": lambda x: float(2.0 ** x),
                "slider_transform_inverse": lambda x: float(math.log(max(x, 1), 2)),
                "step": 1,
            }
            widget_type = NodeSlider
        elif typename in ("MathFunction", "String"):
            widget_type = NodeDisplay
        elif typename == "RGB":
            value = value.append(1)
            widget_type = NodeColor
        elif typename == "Tensor":
            value = "Tensor does not have a usable frontend element!"
            widget_type = NodeDisplay
        elif typename in ["Tuple", "Vector2", "Vector3"]:
            widget_type = NodeMultiWidget
            sub_widgets = []
            value = list(value)
            for index, field in enumerate(type_specs.get("fields", [])):
                widget = self.generate_generic_widget(
                    type_specs=field, value=value[index], name=name, set_value=set_value
                )
                sub_widgets.append(widget)
            specific_params = {"sub_widgets": sub_widgets}

        elif typename in ["Selector", "FixedSelector"]:
            widget_type = NodeRV
            recycle_view_data = create_rv_data(type_specs.get("known_values"))
            text_widget = []
            if typename == "Selector":
                text_widget = [
                    self.generate_generic_widget(
                        type_specs=type_specs["ntype"],
                        value=value,
                        name=name,
                        set_value=set_value,
                    )
                ]
            specific_params = {
                "sub_widgets": text_widget,
                "recycle_view_data": recycle_view_data,
            }
        if widget_type is None:
            raise TypeError(f"Could not resolve typename '{typename}'")

        params = {
            "param_name": name,
            "node_name": self.node_name,
            "value": value,
            "set_value": set_value,
        }
        params.update(specific_params)
        return widget_type(**params)

    def modify_set(self, node_name, param_name, value, value_type: type = int):
        """
        Slightly modified version of the InputInterface.set.
        This allows to define the type of value.
        """
        self.set(node_name, param_name, value_type(value))

    def create_widget(self, name: str, widget: Widget):
        """
        Create a widget Box for one parameter, possible including multiple sub widgets.

        Args:
            name: Parameter name, will be displayed.
            widget: Widget to add, can consist of multiple sub widgets.
        """
        self.widget_dic[name] = widget
        labeled = BoxLayout()
        labeled.add_widget(Label(text=name, size_hint_x=0.2))
        labeled.add_widget(widget)
        self.ids["widgets"].add_widget(labeled)

    def update_observable(self, *args, **kwargs):
        """
        Change the value of a widget, if changes from the graph manager are communicated,
        Args:
            *args:
            **kwargs: Contains a specific widget name and the updated value.
        """

        widget = kwargs["name"]
        value = kwargs["value"]
        self.widget_dic[widget].value = value
