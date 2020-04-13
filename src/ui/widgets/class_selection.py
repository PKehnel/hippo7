from typing import List

import pygtrie
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from ui.graph_manager import InputInterface
from ui.widgets import get_project_root
from ui.widgets.custom_widgets import create_rv_data
from ui.widgets.custom_widgets import RV


class DropDownWidget(BoxLayout):
    def __init__(self, imagenet_dict, wordnet_dict, add_class_to_pool, **kwargs):
        """
        The DDW consist of a text input field(TIF) and a recycle view(RV). They both observe each other.
        If new text is added in the TIF, the suggestions will be displayed in the RV.
        If a item gets selected in the RV it will be set as text in the TIF.

        Args:
            imagenet_dict: char to labels and labels to idx (pygtrie)
            wordnet_dict: char to hyponym and hyponym to idx (pygtrie)
            add_class_to_pool: function to add a new class to the class pool
            **kwargs:
        """
        super(DropDownWidget, self).__init__(**kwargs)
        text_input = AutoCompleteTextInput(
            imagenet_dict=imagenet_dict,
            wordnet_dict=wordnet_dict,
            add_class_to_pool=add_class_to_pool,
        )
        recycle_view_suggestions = RV(data=[], highlight_behavior=False)
        text_input.bind(suggestions=recycle_view_suggestions.update_data)
        recycle_view_suggestions.bind(selected=text_input.update_text)
        self.add_widget(text_input)
        self.add_widget(recycle_view_suggestions)


class AutoCompleteTextInput(TextInput):
    suggestions = ListProperty([])

    def __init__(self, imagenet_dict, wordnet_dict, add_class_to_pool, **kwargs):
        """
        Text input field to add new labels to the class pool via enter.
        Suggestions are stored in the variable suggestions and generated based on the current input
        and matching patterns in the passed dictionaries.

        Args:
            imagenet_dict: Char to labels and labels to idx (pygtrie).
            wordnet_dict: Char to hyponym and hyponym to idx (pygtrie).
            add_class_to_pool: Function to add a new class to the pool (usage via kv file).
            **kwargs:

        """
        super(AutoCompleteTextInput, self).__init__(**kwargs)
        self.wordnet_dict = wordnet_dict
        self.imagenet_dict = imagenet_dict
        self.add_class_to_pool = add_class_to_pool

    def on_text(self, caller, value):
        """
        This runs a callback when the text changes. The functions first clears all suggestions
        and then adds new suggestions based on the input to the suggestions variable.

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
            value: Current text in the text field.
        """
        self.suggestions = []
        value = value.strip()
        if len(value) > 0:
            if self.imagenet_dict.has_node(value):
                self.suggestions = self.suggestions + create_rv_data(
                    self.imagenet_dict.keys(value),
                    color=(1, 0, 1, 1),
                    personal_args={"text_origin": 0},
                )
            if self.wordnet_dict.has_node(value):
                self.suggestions = self.suggestions + create_rv_data(
                    self.wordnet_dict.keys(value),
                    color=(0, 1, 1, 1),
                    personal_args={"text_origin": 1},
                )

    def update_text(self, caller, value: dict):
        """
        Update the text and text_origin value.

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
            value: dict containing the text and the text_origin
        """
        self.text = value["text"]
        text_origin = value["text_origin"]
        self.add_class_to_pool(self.text, text_origin)


class ClassSelection(InputInterface, BoxLayout):
    def __init__(
        self,
        param_name: str,
        node_name: str,
        value: List,
        graph_manager,
        attributes,
        **kwargs,
    ):
        """
        Widget for class selection. This widget consist of 4 sub parts:
         - recycle_view_class_pool: a recycle view (RV) showing all class labels currently in the pool
         - drop_down_widget: a text input field to add new classes to the pool via enter, extend by a RV for displaying suggestions
         - delete button: to remove exactly one entry from the pool
         - clear all button: to remove all except one entry from the pool
         - move entry button: to put a class to the top of the class pool. Ordering is important for the class sampler


        Args:
            param_name: Parameter name.
            node_name: Node name.
            value: List of indexes currently in the class pool.
            graph_manager: Graph Manager.
            attributes: Observable attributes.
            **kwargs:
        """
        super(ClassSelection, self).__init__(graph_manager, **kwargs)
        attributes.subscribe(self.update_observable)
        self.node_name = node_name
        self.param_name = param_name
        self.value = value

        self.load_dicts()
        self.class_pool_rv_format = []
        self.create_class_pools()

        drop_down_widget = DropDownWidget(
            imagenet_dict=self.imagenet_labels,
            wordnet_dict=self.wordnet_labels,
            add_class_to_pool=self.add_class_to_pool,
        )
        self.recycle_view_class_pool = RV(data=self.class_pool_rv_format,)
        self.ids.class_pool.add_widget(self.recycle_view_class_pool)
        self.ids.buttons.add_widget(
            Button(text="Move Entry", on_press=self.move_node_to_top)
        )
        self.ids.buttons.add_widget(
            Button(text="Clear All", on_press=self.reset_class_pool)
        )
        self.ids.buttons.add_widget(
            Button(text="Delete Entry", on_press=self.delete_entry)
        )
        self.ids.class_selection.add_widget(drop_down_widget)

    def create_class_pools(self):
        """
        Initialize the class pool. Currently this information is stored in two locations:
            - self.value (list of class idx currently in the pool)
            - self.class_pool_list_of_dicts (RV data format)

        This is due the internal working of the recycle view that requires a list containing a dictionary.
        """
        transformed_value = list(
            map(lambda x: f"{self.imagenet_classes[x][0]}", self.value)
        )
        self.class_pool_rv_format = create_rv_data(
            transformed_value, personal_args={"text_origin": 0}
        )

    def delete_entry(self, *args):
        """
        Delete the selected (highlighted) entry from the class pool.

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
        """
        if len(self.value) > 1 and self.recycle_view_class_pool.selected:
            label = self.recycle_view_class_pool.selected["text"]
            idx = self.imagenet_labels[label]
            self.value.remove(idx)
            self.set_value()

    def move_node_to_top(self, *args):
        """
        Move the selected node to the the start of the class pool. This is mainly for the class sampler,
        who works position based. (E.g. star sampling taxed node at position [0] as center).

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
        """
        if self.recycle_view_class_pool.selected:
            label = self.recycle_view_class_pool.selected["text"]
            self.delete_entry()
            self.add_class_to_pool(label)

    def reset_class_pool(self, *args):
        """
        Remove all classes except one, since the samples always need an output.
        Currently the one class is fixed to mask.

        Args:
            caller: The caller of the function. Kivy "bind" internal behavior.
        """
        self.value = [643]
        self.class_pool_rv_format = [{"text": "mask"}]
        self.set_value()

    def update_observable(self, *args, **kwargs):
        """
        Observable node pattern. Triggered when the observed value is changed.
        Updates the value in both stored locations.

        Args:
            **kwargs: Dict containing the parameter name and the updated value.
        """
        self.value = kwargs["value"]
        self.create_class_pools()
        self.recycle_view_class_pool.data = self.class_pool_rv_format

    def add_class_to_pool(self, class_label, text_origin=0, index=0):
        """
        Add a list of classes to the class pool.
        Args:
            class_label: Either a single class label or a hyponym.
            text_origin: 0 = single class, 1 = hyponym.
            index: Index at which position to add the class. Defaults to the front of the list.
        """
        if text_origin == 1:
            values = self.wordnet_labels[class_label]
        else:
            values = [class_label]
        for item in values:
            try:
                self.value.insert(index, self.imagenet_labels[item])
                self.class_pool_rv_format.insert(
                    index, {"text": self.imagenet_labels[item]}
                )
            except KeyError:
                print(f"{class_label} is not an existing class")
        self.set_value()

    def set_value(self):
        self.set(node_name=self.node_name, param_name=self.param_name, value=self.value)

    def load_dicts(self):
        """
        For better runtime speed multiple trees and dictionaries are precomputed, mapping between:
         - imagenet_labels: Char to labels and labels to idx (pygtrie).
         - imagenet_classes: Idx to label (dict).
         - wordnet_labels: Char to hyponym and hyponym to idx (pygtrie).
        """
        import json

        path = get_project_root()
        self.imagenet_labels: pygtrie = pygtrie.CharTrie(
            json.load(
                open(
                    path / "assets/imagenet_classes/trie_char_to_name_to_idx.json", "rb"
                ),
            )
        )
        self.imagenet_classes: dict = json.load(
            open(path / "assets/imagenet_classes/dic_idx_to_label.json", "rb"),
            object_hook=key_to_int,
        )

        self.wordnet_labels: pygtrie = pygtrie.CharTrie(
            json.load(
                open(path / "assets/imagenet_classes/hyponym_imagenet.json", "rb")
            )
        )


def key_to_int(x):
    """
    When using json keys are interpreted as strings.
    The functions transforms all keys to int.

    Args:
        x: Dictionary with Int keys stored as String

    Returns:
        Input dictionary with Int keys.
    """
    recovered_dic = {}
    for k, v in x.items():
        recovered_dic[int(k)] = v
    return recovered_dic
