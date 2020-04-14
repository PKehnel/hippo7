from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.togglebutton import ToggleButton

import hippo7_app.hippo7_frontend.graph_manager as manager
from hippo7_app.hippo7_frontend.nodepage import NodePage
from hippo7_app.hippo7_frontend.util import get_color
from hippo7_app.hippo7_frontend.widgets.bpm_widget import BPMManager
from hippo7_app.hippo7_frontend.widgets.bpm_widget import GlobalBPMWidget
from hippo7_app.hippo7_frontend.widgets.custom_widgets import CustomToggleButton


class Hippo7Layout(manager.InputInterface, FloatLayout):

    screen_dic = {}

    def __init__(self, config=None, **kwargs):
        """
        Initialize the main layout of the app, consisting off:
         - Play / Pause Button
         - Print Graph Button
         - Global BPM Widget
         - Navigation bar
         - Custom pages according to the graph from the backend

        For each node in the graph a screen is created and added to the screen manager.

        Args:
            target: Address of the server.
            port: Port from the server.
            config: hippo7.ini containing information like port, btn.color
            **kwargs:
        """
        super(Hippo7Layout, self).__init__(**kwargs)
        self.config = config
        self.graph_manager = manager.GraphManager(
            target=config.get("General", "target"), port=config.get("General", "port")
        )
        self.node_observables = self.graph_manager.node_observables

        self.pipe_navigation: BoxLayout = self.ids["pipe_navigation"]
        self.main_interface_sm: ScreenManager = self.ids.screen_manager_SL

        self.bpm_manager = BPMManager(
            observable_node=self.node_observables["bpm_clock"]["bpm"],
            graph_manager=self.graph_manager,
        )

        self.ids["play_button"].add_widget(
            CustomToggleButton(function=self.graph_manager.change_sending)
        )
        self.ids["presets"].add_widget(
            Button(
                text="Generate Preset",
                on_press=lambda x: self.graph_manager.graph_to_str(),
            )
        )
        self.ids["bpm_widget"].add_widget(
            GlobalBPMWidget(
                bpm_manager=self.bpm_manager,
                observable_node=self.node_observables["bpm_clock"]["bpm"],
                graph_manager=self.graph_manager,
            )
        )
        for node_name, node in self.node_observables.items():
            if node:
                self.create_page(node_name, node)
        self.presets: GridLayout = self.ids["presets"]
        for name, toggle in self.graph_manager.preset_list:
            self.create_preset(name, toggle)

    def create_page(self, node_name, node):
        """
        Create a new page for the node and add a button to reach the node to the navigation bar.
        Args:
            node_name: Name of the node.
            node: Node to add.
        """
        btn = Button(text=node_name, id=node_name)
        main_interface = NodePage(
            node_name=node_name, node=node, graph_manager=self.graph_manager,
        )
        screen = Screen(name=node_name)
        screen.add_widget(main_interface)
        self.screen_dic[node_name] = screen
        btn.bind(on_press=lambda x: self.navigation_btn_function(x))
        btn.background_color = get_color(self.config.get("Color", "btn_background_col"))
        self.pipe_navigation.add_widget(btn)

    def navigation_btn_function(self, btn: Button):
        """
            Action to perform for the navigation bar buttons, telling the screen manger to switch to a specif screen.
            change color of active screen
            @param btn: Navigation Bar Button
            """
        for btn_pipe in self.pipe_navigation.children:
            btn_pipe.background_color = get_color(
                self.config.get("Color", "btn_background_col")
            )
        btn.background_color = get_color(self.config.get("Color", "btn_col"))
        self.main_interface_sm.switch_to(
            screen=self.screen_dic[btn.text], direction="left", duration=0.25,
        )

    def create_preset(self, name: str, toggle: bool):
        """
        Create a Button to activate a preset.
        Args:
            name: Name of the preset.
            toggle: Toggle button or not.
        """
        if toggle:
            btn = ToggleButton(text=name, id=name)
        else:
            btn = Button(text=name, id=name)
        btn.bind(on_press=lambda x: self.preset_btn_function(x))
        self.presets.add_widget(btn)

    def preset_btn_function(self, btn: Button):
        """
        Handle the button press, important for toggle buttons.
        Args:
            btn:
        """
        status = "off" if btn.state == "normal" else "on"
        self.graph_manager.activate_preset(btn.id, status)

    def update_observable(self, *args, **kwargs):
        pass


class Hippo7App(App):
    def build_config(self, config):
        """
        Sets default values for the config. Only applied if no entry exists in the hippo7.ini file.
        Args:
            config: local config file (hippo7.ini)
        """
        config.setdefaults("General", {"target": "localhost", "port": 8000})

    def build(self):
        return Hippo7Layout(config=self.config)
