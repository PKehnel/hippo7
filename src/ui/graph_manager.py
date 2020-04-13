from abc import abstractmethod

from ui.backend_interface import BackendInterface


class Observable(object):
    def __init__(self, name, value, **kwargs):
        """
        Implements the observer pattern. All subscribes to the object will be notified about changes.
        Every call of set_value triggers a notify.

        Args:
            name: Name of the observable.
            value: Only value that will trigger notify messages. Changes to other values are ignored.
            **kwargs: Other arguments that will be propagated alongside with name and value.
        """
        self.name = name
        self.value = value
        self.params = {"name": self.name, "value": self.value}
        self.params.update(**kwargs)
        self.__callbacks = []

    def subscribe(self, callback):
        """
        Subscribe to this node to get notified when the value gets changed.

        Args:
            callback: Function that will be called on change, with the updated value as parameter.
        """
        self.__callbacks.append(callback)

    def unsubscribe(self, callback):
        self.__callbacks.remove(callback)

    def set_value(self, value):
        """
        Changes local value and triggers notify.
        Values have to be changed using this method, otherwise notify will not be called.

        Args:
            value: The new Value.
        """
        self.params["value"] = value
        self.value = value
        self.notify()

    def notify(self):
        """
        Notify all subscribed members about an updated value.
        """
        for member in self.__callbacks:
            member(**self.params)


class GraphManager:
    def __init__(self, target="localhost", port=8000):
        """
        The GraphManager handles the communication between frontend and backend. It stores the graph and propagates
        changes to listeners in the frontend as well as the backend.

        By default he retrieves a graph from the backend and turns the nodes into Observables and stores them
        in node_param dictionary.
        """
        self.node_observables = {}
        self.node_regulations = {}
        self._sending = True
        self._changed_nodes = []
        self.backend = BackendInterface(target, port)

        self.graph, self.preset_list, self.presets = self.backend.get_graph()
        self.node_behaviours = self.backend.get_node_behaviours()
        self.map_types_to_nodes()

        for name, node in self.graph.items():
            node_inner = {}
            for parameter, value in node.items.items():
                node_inner[parameter] = Observable(
                    name=parameter, value=value, node_name=name
                )
            self.node_observables[node.name] = node_inner

    def map_types_to_nodes(self):
        """
        Map the node behaviour to the nodes in the graph.
        """
        for name, node in self.graph.items():
            self.node_regulations[name] = self.node_behaviours[node.node_type]

    def set(self, node_name, param_name, value):
        """
        Propagates changes to the observable node and the backend.
        Args:
            node_name: Node name (id).
            param_name: The parameter of the node that changed.
            value: The new value for the parameter.
        """
        self.graph[node_name].items[param_name] = value
        node = self.graph[node_name]
        self.node_observables[node_name][param_name].set_value(value)
        if self._sending:
            self.backend.set_node_value(node_name=node.name, node_dict=dict(node.items))
        else:
            self._changed_nodes.append(node.name)

    def graph_to_str(self):
        """
        Print the graph in human readable form. This can be directly used as preset.
        """
        for node in self.graph.values():
            for item, value in node.items.items():
                if isinstance(value, str):
                    value = "'" + value + "'"
                print(f"'{node.name}.{item}':{value},")

    def change_sending(self):
        """
        To allow making multiple changes before sending, this can cache changes and send them all at once.
        """
        self._sending = not self._sending
        if self._sending:
            for node_name in self._changed_nodes:
                node = self.graph[node_name]
                self.backend.set_node_value(
                    node_name=node.name, node_dict=dict(node.items)
                )
            self._changed_nodes = []

    def activate_preset(self, id: str, status: str):
        """
        Trigger all changes defined in a preset.
        Args:
            id: Id of a preset.
            status: "On" / "Off" used for toggle presets.
        """
        for preset in self.presets[id]:
            if preset[0] in ["values", status]:
                for key, value in preset[1].items():
                    splitted_key = key.split(".")
                    node_name = splitted_key[0]
                    param_name = splitted_key[1]
                    self.set(node_name, param_name, value)


class InputInterface:
    def __init__(self, graph_manager: GraphManager = None, **kwargs):
        """
        "ABC" for input interfaces. They all need access to the same graph_manager to have access on the same nodes
        Args:
            graph_manager: Manages the graph, propagates changes in all directions.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.graph_manager = graph_manager

    @abstractmethod
    def update_observable(self, *args, **kwargs):
        pass

    def set(self, node_name, param_name, value):
        """
        Direct forwarding to the set of graph_manager.
        """
        self.graph_manager.set(node_name, param_name, value)
