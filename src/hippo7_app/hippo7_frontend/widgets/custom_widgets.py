from functools import partial
from typing import List

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.properties import DictProperty
from kivy.properties import ListProperty
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.togglebutton import ToggleButton


class CustomButton(Button):
    text = StringProperty("custom")
    start_value = NumericProperty(1.0)

    def __init__(
        self,
        function=lambda x: x,
        start_value_function=None,
        function_call_back=None,
        hold_down=False,
        **kwargs
    ):
        """
        Custom Button for binary operations with predefined second operators.

        Args:
            function: Any sort of binary operation.
            start_value_function: If operand comes from a function.
            function_call_back: Function to call with result value.
            hold_down: If on hold down the operation will be triggered repeatedly.
            kwargs:
        """
        super(CustomButton, self).__init__(**kwargs)
        self.function = function
        self.start_value_function = start_value_function
        self.function_call_back = function_call_back
        self.hold_down = hold_down

    def start_event(self):
        """
        Trigger a function. If the button stays on hold_down execute the function continuously.
        """
        self.check_func_init()
        self.execute_func()
        if self.hold_down:
            self.button_event = Clock.schedule_interval(self.execute_func, 1 / 5)

    def check_func_init(self):
        """
        If a function is given, its used for initialization.
        """
        if self.start_value_function:
            self.start_value = float(self.start_value_function())

    def execute_func(self, *args):
        """
        Get the start value from the specified function and trigger the callback function.
        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
        """
        self.start_value = self.function(self.start_value)
        self.function_call_back(self.start_value)

    def cancel_btn_event(self):
        """
        On button release stop the current event.
        """
        if self.hold_down:
            self.button_event.cancel()


class NodeWidget(BoxLayout):
    def __init__(self, node_name, param_name, value, set_value, **kwargs):
        """
        Standard for all NodeWidgets handling the initialization.
        Also providing a simplified set function, where only the value has to be given.

        Args:
            node_name: current node name
            param_name: adjustable param
            value: current value
            set_value: function to call on value change
            kwargs:
        """
        super(NodeWidget, self).__init__(**kwargs)
        # components like the switch need a standard value
        # if that value gets overwritten at initialization, set_value is trigger, but not existing
        # changing the order of set_value and value, doesnt resolve the issue, since set value cant be triggered,
        # before the widget is actually added to the page.
        # -> placeholder set_value for init changes

        self.set_value = lambda *args: args
        self.node_name = node_name
        self.param_name = param_name
        self.value = value
        self.set_value = set_value

    def simple_set_value(self, value, *args):
        """
        Simplified version of set value, that only requires the new value.

        Args:
            value: Updated value.
            caller: The caller of the function. Kivy "bind" internal behavior.
        """
        self.value = value
        self.set_value(self.node_name, self.param_name, self.value)


class NodeColor(NodeWidget):
    def __init__(self, **kwargs):
        """
        Color Wheels from kivy garden are the standard representation for RGB colors.

        Args:
            **kwargs: See parent class.
        """
        super(NodeColor, self).__init__(**kwargs)


class NodeDisplay(NodeWidget):
    value = StringProperty("Something went wrong")

    def __init__(self, **kwargs):
        """
        Text Fields are the standard representation for strings.

        Args:
            **kwargs: See parent class.
        """
        super(NodeDisplay, self).__init__(**kwargs)


class NodeRV(NodeWidget):

    data = ListProperty([{"text": "empty"}])
    value = StringProperty("Something went wrong")

    def __init__(self, recycle_view_data: List[dict], sub_widgets, **kwargs):
        """
        RV are the standard representation for selectable lists.
        They can be combined with a second widget, normally a NodeDisplay.

        Args:
            **kwargs: See parent class.
        """
        super(NodeRV, self).__init__(**kwargs)
        self.data = recycle_view_data
        recycle_view = RV(data=self.data)
        recycle_view.bind(
            selected=lambda caller, value: self.simple_set_value(value=value["text"])
        )
        self.bind(value=recycle_view.update_selected)
        self.add_widget(recycle_view)
        for widget in sub_widgets:
            self.bind(value=lambda caller, value: widget.simple_set_value(value=value))
            self.add_widget(widget)


class NodeSwitch(NodeWidget):
    value = BooleanProperty(True)

    def __init__(self, **kwargs):
        """
        Switches are the standard representation for bool values.

        Args:
            **kwargs: See parent class.
        """
        super(NodeSwitch, self).__init__(**kwargs)


class NodeSlider(NodeWidget):

    value = NumericProperty(1)

    def __init__(self, min_value=0, max_value=15, step=1, **kwargs):
        """
        Slider are the standard representation for int and float values.

        Args:
            min_value: Left end of slider.
            param max_value: Right end of slider.
            param step: Step size, needs to be int for integer sliders.
            kwargs: See parent class.
        """
        # self.root_set_value = kwargs['set_value']
        # kwargs['set_value'] = self.set_value
        super(NodeSlider, self).__init__(**kwargs)
        self.min = min_value
        self.max = max_value
        self.step = step
        self.slider = self.ids["slider"]
        # self.slider.value = kwargs['value']

    def set_value(self, node_name, param_name, value):
        self.slider.value = self.slider_transform_inverse(value)
        self.root_set_value(node_name, param_name, value)


class NodeMultiWidget(NodeWidget):
    value = ListProperty([])

    def __init__(self, sub_widgets: List = None, **kwargs):
        """
        MultiWidgets consist of a parent widget observing the value and a number of closely related child widgets.
        They all are based on the same value that they observe and modify.

        Args:
            sub_widgets: List of child widgets
            **kwargs: See parent class.
        """
        super(NodeMultiWidget, self).__init__(**kwargs)
        if sub_widgets is None:
            sub_widgets = []
        self.sub_widgets = sub_widgets
        for i, widget in enumerate(sub_widgets):

            def set_function(node_name, param_name, inner_value, i=0):
                self.value[i] = inner_value
                return self.set_value(node_name, param_name, tuple(self.value))

            widget.set_value = partial(set_function, i=i)
            self.add_widget(widget)

        self.bind(value=self.set_subwidgets)

    def set_subwidgets(self, caller, value):
        """
        Update the value of the child widgets, since they dont follow the observer pattern them self.

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
            value: Updated value of the parent widget.
        """
        for i, widget in enumerate(self.sub_widgets):
            widget.value = value[i]


class RV(RecycleView):
    selected = DictProperty()

    def __init__(self, data: List[dict], highlight_behavior: bool = True, **kwargs):
        """
        Basic RecycleView used for displaying and selecting data.

        Args:
            data: List of dictionaries that need to contain the attribute 'text'. The value of text will be displayed.
            highlight_behavior: indicate the highlight_behavior
            **kwargs:
        """
        super(RV, self).__init__(**kwargs)
        self.data = data
        self.highlight_behavior = highlight_behavior

    def update_data(self, caller, data):
        self.data = data

    def update_selected(self, caller, value):
        """
        If a node exists with the same internal value as the send value, set this node to selected.
        Args:
            caller: caller of the function
            value: value to highlight
        """
        for index, node in enumerate(self.data):
            if value == node["text"]:
                self.layout_manager.select_node(index)


def create_rv_data(data: list, personal_args: dict = {}, color=(1, 0, 1, 1)) -> list:
    """
    The Kivy recycle view takes data in a specific format. A list containing dictionaries that requires the entry "text",
    where the displayed value will be stored. There is another set of possible arguments (e.g color),
    and custom arguments are allowed.

    Args:
        data: Data to display in a list.
        personal_args: Stored with each entry.
        color: Sets the displayed color value.

    Returns: a List in RV format

    """
    rv_data_format = []
    if color not in personal_args:
        personal_args["color"] = color
    for index, item in enumerate(data):
        transformed_data = personal_args.copy()
        transformed_data["text"] = item
        rv_data_format.append(transformed_data)
    return rv_data_format


class CustomToggleButton(ToggleButton):
    def __init__(
        self,
        function,
        function_on_release=None,
        text_options=["Play", "Pause"],
        **kwargs
    ):
        """
        A Toggle Button that changes the displayed text.

        Args:
            function: Function to execute on button press.
            function_on_release: Second function triggered on "release press", only if specified.
            text_options: Text to display for both button states.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.text_options = text_options
        self.function = function

        self.function_on_release = (
            function if not function_on_release else function_on_release
        )
        self.button_state = False
        self.text = self.text_options[0]

    def on_press(self):
        """
        Trigger the function and update the displayed text.
        """
        self.button_state = not self.button_state
        if self.button_state:
            self.function()
            self.text = self.text_options[1]
        else:
            self.function_on_release()
            self.text = self.text_options[0]


# The following two classes are directly copied from:
# https://kivy.org/doc/stable/api-kivy.uix.recycleview.html
# They handle the selecting behavior.


class SelectableRecycleBoxLayout(
    FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout
):
    """ Adds selection and focus behaviour to the view. """


class SelectableLabel(RecycleDataViewBehavior, Label):
    """ Add selection support to the Label """

    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        """ Add selection on touch down """
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """ Respond to the selection of items in the view. """
        self.selected = is_selected
        if is_selected:
            rv.selected = rv.data[index]
            if not rv.highlight_behavior:
                rv.layout_manager.clear_selection()
