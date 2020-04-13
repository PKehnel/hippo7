from unittest import mock
from unittest.mock import MagicMock

from renderer.pipeline.json import graph_to_json
from renderer.pipeline.json import json_to_graph
from renderer.pipeline.json import node_behaviour_to_json
from renderer.pipeline.json import node_behaviours_to_json
from renderer.pipeline.json import node_to_json


def test_to_json(node_factory):
    cls, params = node_factory
    node = cls(**params)
    inverse_params = {v: k for k, v in params.items()}
    print(node_to_json("testname", node, inverse_params))


def shader(*args, **kwargs):
    s = MagicMock()
    s.draw = lambda x, y: None
    return s


@mock.patch("renderer.pipeline.nodes.gloo.Program", shader)
def test_graph(graph):
    jsonified_graph = graph_to_json(graph)
    graph_json = jsonified_graph.json()
    regraphed_graph = json_to_graph(graph_json)
    regraphed_graph.draw()
    for g0, g1 in zip(regraphed_graph.presets.items(), graph.presets.items()):
        assert g0[0] == g1[0]
        for g01, g11 in zip(g0[1].values.items(), g1[1].values.items()):
            if isinstance(
                g11[1], tuple
            ):  # Tiny hack to fix the fact that tuples can not be jsonified properly
                g11 = g11[0], list(g11[1])
            assert g01 == g11


def test_node_behaviour(node_factory):
    cls, params = node_factory
    print(node_behaviour_to_json(cls.__name__, cls).json())


def test_node_behaviours():
    behaviours = node_behaviours_to_json()
    print(behaviours.json())
