import sys
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from hippo7_app.hippo7_backend import graph_setup

sys.path.append("src/")  # NOQA
sys.path.append("../src/")  # NOQA
import hippo7_app.hippo7_backend.pipeline.nodes as  N
from hippo7_app.hippo7_backend.opengl.render import WindowManager
from hippo7_app.hippo7_backend.pipeline.graph import RenderGraph
from hippo7_app.hippo7_backend.pipeline.json import get_classes_from_module


@pytest.fixture(autouse=True, scope="session")
def mock_window():
    """ This function is used to initialize the WindowManger Singleton. Note that it is
    called automatically once for the module. """
    fake_window = Mock()
    fake_window.width = 100
    fake_window.height = 100
    fake_window.clear = Mock()
    WindowManager.bind_window(fake_window)


def mock_shader(*args, **kwargs):
    fake_shader = Mock()
    fake_shader.draw = Mock()
    fake_shader.attach = Mock()
    fake_shader.get_link = lambda: fake_shader
    return fake_shader


def mock_biggan(*args, **kwargs):
    return Mock()


@pytest.fixture(
    params=get_classes_from_module(N, N.Node).keys(),
    ids=get_classes_from_module(N, N.Node).keys(),
)
def node_name(request) -> str:
    return request.param


@pytest.fixture(
    params=get_classes_from_module(N, N.Node).items(),
    ids=get_classes_from_module(N, N.Node).keys(),
    scope="session",
)
@mock.patch("hippo7_app.hippo7_backend.pipeline.nodes.gloo.Program", mock_shader)
def node_factory(request) -> Tuple[type, dict]:
    name, cls = request.param
    params = {}
    if name == "DrawNode":
        params = {"window": N.Window().window, "shader": mock_shader()}
    elif name == "GeometryController":
        params = {"shader": N.Shader().shader, "window_size": N.Window().size}
    elif name == "SampleManager":
        params = {"sampler": MagicMock()}
    return cls, params


@pytest.fixture
def node(node_factory):
    cls, params = node_factory
    return cls(**params)


@pytest.fixture(
    params=[
        graph_setup.no_gan_setup,
        graph_setup.simple_gan_setup,
        graph_setup.complex_gan_setup,
    ],
    ids=["setup_no_gan", "simple_gan_setup", "complex_gan_setup"],
)
@mock.patch("hippo7_app.hippo7_backend.pipeline.nodes.gloo.Program", mock_shader)
def graph(request):
    g = request.param()
    return g


def add_default_nodes(g: RenderGraph, bpm=120):
    g.add_node(N.Window(), name="window")
    g.add_node(N.BpmClock(bpm=bpm), name="bpm_clock")
    return g
