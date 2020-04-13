from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import numpy as np
import torch
from pydantic import BaseModel

import renderer.pipeline.nodes as N
from renderer.pipeline.graph import Node
from renderer.pipeline.graph import NodeInput
from renderer.pipeline.graph import NodeOutput
from renderer.pipeline.graph import Preset
from renderer.pipeline.graph import RenderGraph
from renderer.pipeline.graph import TogglePreset
from renderer.pipeline.types import Ntype


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


def get_classes_from_module(module, filter_class=None):
    module_classes = {k: v for k, v in module.__dict__.items() if isinstance(v, type)}
    if filter_class:
        module_classes = {
            k: v for k, v in module_classes.items() if filter_class in v.__bases__
        }
    return module_classes


node_classes = get_classes_from_module(N, filter_class=Node)


def ntype_to_dict(ntype, check_null=True):
    if check_null and ntype is None:
        raise TypeError("No Ntype! Not serializable!")
    if isinstance(ntype, Ntype):
        fields = {
            k: ntype_to_dict(v, check_null=False) for k, v in ntype.__dict__.items()
        }
        fields["typename"] = ntype.__class__.__name__
        return fields
    elif isinstance(ntype, str):
        return ntype
    elif isinstance(ntype, Iterable):
        return [ntype_to_dict(x) for x in ntype]
    else:
        return ntype


def node_behaviour_to_json(name, node_class):
    fields = node_class.__dict__
    inputs = {
        fieldname: ntype_to_dict(field.ntype)
        for fieldname, field in fields.items()
        if isinstance(field, NodeInput)
    }
    outputs = {
        fieldname: ntype_to_dict(field.ntype)
        for fieldname, field in fields.items()
        if isinstance(field, NodeOutput)
    }
    return NodeBehaviourMessage(typename=name, inputs=inputs, outputs=outputs)


def node_behaviours_to_json():
    behaviours = {}
    for name, node_class in node_classes.items():
        behaviours[name] = node_behaviour_to_json(name, node_class)
    return AllBehavioursMessage(nodes=behaviours)


def node_to_json(name, node: Node, inverse_params):
    flat_inputs = {}
    for key, value in node.inputs().items():
        in_edge = value.get_link()
        if isinstance(in_edge, NodeOutput):
            flat_inputs[key] = inverse_params[in_edge]
        else:
            if isinstance(in_edge.value, np.ndarray):
                in_edge_value = in_edge.value.tolist()
            elif isinstance(in_edge.value, torch.Tensor):
                in_edge_value = in_edge.value.tolist()
            else:
                in_edge_value = in_edge.value
            flat_inputs[key] = in_edge_value
    return NodeMessage(id=name, typename=node.__class__.__name__, inputs=flat_inputs)


def preset_to_json(preset: Union[Preset, TogglePreset]):
    if isinstance(preset, TogglePreset):
        return TogglePresetMessage(on=preset.on, off=preset.off)
    else:
        return PresetMessage(values=preset.values)


def graph_to_json(graph: RenderGraph):
    inverse_outputs = {v: k for k, v in graph.outputs().items()}
    node_messages = []
    for name, node in graph.nodes().items():
        node_messages.append(node_to_json(name, node, inverse_outputs))
    preset_messages = {
        name: preset_to_json(preset) for name, preset in graph.presets.items()
    }
    return GraphMessage(
        draw_node=graph.draw_node(), nodes=node_messages, presets=preset_messages
    )


def json_to_node(node_message):
    name = node_message.id
    node = node_classes[node_message.typename]()
    return name, node, node_message.inputs


def json_to_preset(preset: Union[PresetMessage, TogglePresetMessage]):
    if isinstance(preset, TogglePresetMessage):
        return TogglePreset(preset.on, preset.off)
    else:
        return Preset(preset.values)


def json_to_graph(json_str):
    json_graph = GraphMessage.parse_raw(json_str)
    graph = RenderGraph()
    inputs_to_resolve = []
    for node_message in json_graph.nodes:
        name, node, inputs = json_to_node(node_message)
        graph.add_node(node, name=name)
        inputs_to_resolve.append((name, inputs))
    for name, inputs in inputs_to_resolve:
        for variable, node_input in inputs.items():
            try:
                keys = node_input.split(".")
                node_input = getattr(graph[keys[0]], keys[1])
            except (
                KeyError,
                AttributeError,
            ):  # If it can't resolve the node_input to some output, treat it as a default value
                pass
            graph[name].bind(variable, node_input)
    graph.set_draw_node(json_graph.draw_node)
    for name, preset in json_graph.presets.items():
        graph.add_preset(name, json_to_preset(preset))
    return graph
