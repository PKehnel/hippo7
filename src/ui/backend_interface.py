import socket
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import requests
from pydantic import BaseModel

"""
Code duplicate with backend, defining message types for validation
https://pydantic-docs.helpmanual.io/
"""


class NodeMessage(BaseModel):
    id: str
    typename: str
    inputs: dict


class PresetMessage(BaseModel):
    values: dict


class TogglePresetMessage(BaseModel):
    on: dict
    off: dict


class GraphMessage(BaseModel):
    draw_node: str
    nodes: List[NodeMessage]
    presets: Dict[str, Union[PresetMessage, TogglePresetMessage]]


class NodeBehaviourMessage(BaseModel):
    typename: str
    inputs: dict
    outputs: dict


class AllBehavioursMessage(BaseModel):
    nodes: Dict[str, NodeBehaviourMessage]


class SetValueMessage(BaseModel):
    id: str
    values: Dict[str, Any]


class BackendInterface:
    def __init__(self, address="lex", port=8000, **kwargs):
        """
        Establish to a server at the given address and port.
        """
        self.connect(address, port)

    def set_node_value(self, node_name, node_dict: dict):
        """
        Send a message to the backend, with the name and values of a updated node.

        Args:
            node_name: Name of the modified node.
            node_dict: Parameter names with values.
        """

        message = SetValueMessage(id=node_name, values=node_dict)
        requests.post(self.channel, data=message.json(), params="set_node")

    def get_node_behaviours(self) -> dict:
        """
        Request the behaviour for all existing nodes from the backend.

        Returns:
            Dictionary mapping from node to type and behaviour.

        """
        response = requests.get(self.channel, params="get_behaviours")
        response = AllBehavioursMessage.parse_raw(response.content)
        return response.nodes

    def get_graph(self) -> dict:
        """
        Request the currently used graph from the backend.

        Returns:
            A decoded graph consisting of nodes.

        """
        response = requests.get(self.channel, params="get_graph")
        return json_to_graph(response.content)

    def connect(self, address, port):
        """
        Connect to a server with a given address & port.
        """
        address = socket.getfqdn(address)
        self.channel = "http://" + str(address) + ":" + str(port)


def get_node(node_message):
    id = node_message.id
    typename = node_message.typename
    inputs = node_message.inputs
    return id, typename, inputs


def json_to_graph(json_str: str) -> (dict, list, dict):
    """
    Transform a string (json message) containing the backend graph and presets to
    a graph and presets in frontend format.
    Args:
        json_str: Backend graph as json string

    Returns:
        Dictionary with all nodes and their inputs,
        a list of preset names and their toggle behavior and a dict with the actual preset values.

    """
    json_graph = GraphMessage.parse_raw(json_str)
    node_dict = {}
    inputs_to_resolve = []

    for node_message in json_graph.nodes:
        name, node, inputs = get_node(node_message)
        node = Node(name=name, node_type=node)
        inputs_to_resolve.append((name, inputs))
        node_dict[name] = node

    for name, inputs in inputs_to_resolve:
        for variable, node_input in inputs.items():
            try:
                keys = node_input.split(".")
                node_id, node_output = keys[0], keys[1]
                node_dict[name].inputs[variable] = (node_id, node_output)
            except (
                KeyError,
                IndexError,
                AttributeError,
            ):  # If it can't resolve the node_input to some output, treat it as a default value
                node_dict[name].items[variable] = node_input

    presets = {}
    preset_list = []
    for name, preset in json_graph.presets.items():
        toggle = isinstance(preset, TogglePresetMessage)
        preset_list.append((name, toggle))
        presets[name] = preset
    return node_dict, preset_list, presets


class Node:
    def __init__(self, name, node_type, items: dict = None, inputs: dict = None):
        """
        The node class represents the backend graph in the frontend.
        The connection of different nodes is shown in the inputs.
        Args:
            name: Node Name (unique).
            node_type: Node type can be shared by multiple nodes and clarifies the behavior.
            items: Parameters and values of the node.
            inputs: Incoming connections from other nodes. Mapping a parameter to a node and a node output.
        """
        if items is None:
            items = {}
        self.node_type = node_type
        self.name = name
        self.items = items
        self.inputs = {} if inputs is None else inputs

    def __repr__(self):
        return f"{self.name}<{self.node_type}>: {self.items}"
