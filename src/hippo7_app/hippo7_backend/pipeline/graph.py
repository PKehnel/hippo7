from abc import abstractmethod
from copy import copy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import hippo7_app.hippo7_backend.pipeline.types as T


class DefaultValue:
    """
    A default value is a type of Input for a Node that is not another NodeOutput but a
    fixed value instead. It can be changed from the frontend.
    """

    def __init__(self, value, ntype=None):
        """ Initialization of a DefaultValue.

        Args:
            value: The value the DefaultValue should return.
            ntype: The ntype of the value(possible types are specified in the types module)
        """
        self.ntype = ntype
        self.value = value

    def _get(self, frame):
        return self.value

    def __repr__(self):
        return self.value.__repr__()


class NodeOutput:
    """
    The NodeOutput is the container class for all output edges of a Node and should be
    set as a class attribute in the Node. It is later used as an connection for the
    NodeInputs.
    """

    def __init__(
        self, value: Any = None, attached_node: object = None, ntype: T.Ntype = None
    ):
        """ The init method.

        Args:
            value: The inital default value for the output.
            attached_node: The node that this NodeOutput is a part of.
            ntype: The ntype specifies which kind of output this edge represents.
        """
        super().__init__()
        if ntype is None and value is not None:
            ntype = T.Ntype.type_from_value(value)
        if value is None and ntype is not None:
            value = ntype.default_value()
        self.ntype = ntype
        self.value = value
        self.attached_node = attached_node

    def _get(self, frame: int) -> Any:
        if frame > self.attached_node._frame:
            self.attached_node.trigger_update(frame)
        return self.value

    def _set(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"Value: {self.value}, Node: {self.attached_node}"

    def attach(self, attached_node: object):
        """ A function to set the attached node at a later date than initialization.

        Args:
            attached_node: A node object.
        """
        self.attached_node = attached_node


class NodeInput:
    """
    The NodeInput represents an input for a node and should be set as a class attribute
    in the Node. It is later either connected with a NodeOutput or is represented by a
    DefaultValue.
    """

    def __init__(
        self, value: Any = None, ntype: T.Ntype = None, pre_bind_hook: Callable = None
    ):
        """ The init method.

        Args:
            value: The default value for the node. It does not need to be set if it can
                    be inferred by the ntype.
            ntype: The ntype specifies which kind of input this node represents. It is
                    important that this is set correctly because it has relevancy for
                    the frontend representation.
            pre_bind_hook: A function that can be additionally specified to run before
                            a DefaultValue assignment to fix certain issues with the
                            input(For example if there exists an easier representation
                            for the frontend).
        """
        super().__init__()
        if pre_bind_hook:
            self.pre_bind_hook = pre_bind_hook
        if ntype is None and value is not None:
            ntype = T.Ntype.type_from_value(value)
        if value is None and ntype is not None:
            value = ntype.default_value()
        self.ntype = ntype
        self.device = "cpu"
        self._set_default(value)

    def _get(self, frame: int) -> Any:
        return self.input._get(frame)

    def bind(self, node_output: Union[NodeOutput, Any]):
        """ Binds either a NodeOutput or a Value to this NodeInput. This way if the node
        gets called the correct value can be evaluated.

        Args:
            node_output: Either a NodeOutput or a valid value that can be packed into a
                        DefaultValue.

        """
        if hasattr(node_output, "attach"):  # Check if it is a NodeOutput
            self.input = node_output
        elif isinstance(node_output, Node):
            raise TypeError(
                "You are trying to bind a Node to this Input, but it should be a NodeOutput instead!"
            )
        else:
            self._set_default(node_output)

    def get_link(self) -> Union[NodeOutput, DefaultValue]:
        """ Returns the bound edge.

        Returns:
            The NodeOutput or DefaultValue assigned to ths node. Note that in case of a
            DefaultValue, the value is represented as is before the pre_bind_hook
            is applied.

        """
        if hasattr(self, "pre_bind_input"):
            return DefaultValue(self.pre_bind_input)
        return self.input

    def __repr__(self) -> str:
        return f"{self.input.value} @ {id(self.input)}"

    def to_device(self, device: str):
        self.device = device

    def _set_default(self, value):
        if hasattr(self, "pre_bind_hook"):
            self.pre_bind_input = value
            value = self.pre_bind_hook(self, value)
        value = self.ntype.cast(value)
        self.input = DefaultValue(value)


class Node(object):
    """
    The Node is the the base class for all nodes that make up a part of a RenderGraph.
    To create a new Node, the Inputs should be specified as class attributes using
    NodeInputs, the Outputs similarly as NodeOutputs. Additionally the update() method
    needs to be implemented, which will control the update of the outputs based on the
    inputs. It will be called every frame.
    """

    def __new__(cls, *args, **kwargs):
        """
        This method exists to replace the class attributes with the proper instance
        attributes at runtime.
        """
        _in_edges = {}
        _out_edges = {}
        instance = super().__new__(cls)
        for k, v in cls.__dict__.items():
            if isinstance(v, NodeInput):
                _in_edges[k] = copy(v)
                _in_edges[k].ntype = v.ntype.copy()

            if isinstance(v, NodeOutput):
                _out_edges[k] = copy(v)
                _out_edges[k].attach(instance)

        object.__setattr__(instance, "_in_edges", _in_edges)
        object.__setattr__(instance, "_out_edges", _out_edges)
        return instance

    def __init__(self, **kwargs: dict):
        """
        The init method sets all input edges to values based on the **kwargs.
        """
        self._frame = 0
        self.device = "cpu"
        for k, v in kwargs.items():
            self.bind(k, v)

    def trigger_update(self, frame: int):
        """
        The trigger_update method guarantees that update will not be called more than
        once per frame. It should be called instead of update() directly.
        """
        if frame > self._frame:
            self._frame = frame
            self.update()

    @abstractmethod
    def update(self):
        """
        Update needs to be overwritten by the children class. It should contain all
        computations based on some input and their assignments to the proper outputs.
        """
        pass

    def __setattr__(self, key, value):
        """
        The setattr method overwrites the standard way to assign values, which allows
        much nicer syntax for the update method.
        """
        edge = self._out_edges.get(key, None)
        if edge:
            edge._set(value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, item):
        """
        The gettattr method is changed to be able to treat the inputs and outputs as
        proper values and not as their managing objects.
        """
        edge = super(Node, self).__getattribute__("_in_edges").get(item, None)
        if edge:
            return edge._get(self._frame)
        edge = super(Node, self).__getattribute__("_out_edges").get(item, None)
        if edge:
            return edge
        return super(Node, self).__getattribute__(item)

    def inputs(self) -> Dict[str, NodeInput]:
        """
        Returns all inputs.
        """
        return self._in_edges

    def outputs(self):
        """
        Returns all outputs.
        """
        return self._out_edges

    def bind(self, key, value):
        """
        Binds either a value or NodeOutput to an input edge, which allows it to be
        either update-called in the update process or changed from the frontend.
        """
        self._in_edges[key].bind(value)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} - In:<{self.inputs()}> - Out:<{self.outputs()}>"
        )

    def to_device(self, device):
        """
        Certain nodes interact heavily with the GPU, but not all machines support those.
        You can use to_device to switch the computations/data to some other device.
        (In most cases this is either 'cpu' or 'cuda')
        """
        self.device = device
        for inp in self._in_edges.values():
            inp.to_device(device)


class Preset:
    """
    A preset is a preset of values for DefaultValues to quickly change the current output
    of the RenderGraph. In essence, it allows us to change multiple values at once, toggle
    values and similar things.
    """

    def __init__(self, values: Dict[str, Any]):
        """ Initialize a preset with a dict of values that should be changed all together.

        Args:
            values: A dict of values where each entry is in the form of
                    "nodename.fieldname": value
        """

        self.values = values

    def apply(self, nodes: Dict[str, Node]) -> Dict[str, Any]:
        """ Applies the changes of the preset to the nodes.

        Args:
            nodes: The node dict of a graph.

        Returns:
            The changed values.
        """

        for key, value in self.values.items():
            key_splitted = key.split(".")
            nodes[key_splitted[0]].bind(key_splitted[1], value)
        return self.values


class TogglePreset(Preset):
    """
    A preset is a preset of values for DefaultValues to quickly change the current output
    of the RenderGraph. In essence, it allows us to change multiple values at once, toggle
    values and similar things.
    In comparison to the normal Preset, the TogglePreset has both an on and off preset
    to simulate a toggling option(basically allow the activation/deactivation of an effect).
    """

    def __init__(self, on: Dict[str, Any], off: Dict[str, Any]):
        """ Initialize a preset with two dicts of values that should be toggled. (The
        first call starts with the on dict)

        Args:
            on: A dict of values where each entry is in the form of
                    "nodename.fieldname": value
            off: A dict of values where each entry is in the form of
                    "nodename.fieldname": value
        """

        super().__init__(dict(on, **off))
        self.on = on
        self.off = off
        self.active = False

    def apply(self, nodes: Dict[str, Node]) -> Dict[str, Any]:
        """ Applies the changes of the preset to the nodes.

        Args:
            nodes: The node dict of a graph.

        Returns:
            The changed values.
        """
        values = self.off if self.active else self.on
        self.active = not self.active
        for key, value in values.items():
            key_splitted = key.split(".")
            nodes[key_splitted[0]].bind(key_splitted[1], value)
        return values


class RenderGraph:
    """
    The RenderGraph is the data structure containing all nodes that make up the
    frame-generation process. Nodes can be added using graph.add_node(node, name). The
    frame-generation process is triggered by calling graph.draw(), where the whole
    graph gets executed from the back, e. g. it starts with the last node(usually a
    DrawNode) and then calls all other nodes by referencing their inputs.
    """

    def __init__(self):
        self.current_frame = 0
        self._nodes: Dict[str, Node] = {}
        self._inputs = {}
        self._outputs = {}
        self._draw_node = None
        self.device = "cpu"
        self.presets = {}

    def draw(self):
        """
        Draws a single frame.
        """
        self.current_frame += 1
        self._draw_node.trigger_update(self.current_frame)

    def draw_without_update(self):
        """
        Draws without updating any of the values. This results in the same output if
        called consecutively.
        """
        self._draw_node.update()

    def add_node(self, node: Node, name: str):
        """ Adds a node to the graph. It is more advisable to use the dict setitem
        syntax instead. (Example: graph[name] = node)

        Args:
            node: The node to be added.
            name: The name or id the node should be referenced as.

        """
        self._nodes[name] = node
        self._inputs.update({f"{name}.{k}": v for k, v in node._in_edges.items()})
        self._outputs.update({f"{name}.{k}": v for k, v in node._out_edges.items()})

    def inputs(self) -> dict:
        """ Returns all NodeInputs for all Nodes.

        Returns:
            All NodeInputs for the graph, so this method does not differentiate
            between Inputs bound to DefaultValues or NodeOutputs.

        """
        return self._inputs

    def outputs(self) -> dict:
        """ Returns all NodeOutputs for all Nodes.

        Returns:
            All NodeOutputs existing in the graph as a dict.

        """
        return self._outputs

    def nodes(self) -> Dict[str, Node]:
        """ Returns all nodes from the graph.

        Returns:
            All Nodes as a dict, mapping the name to the node.

        """
        return self._nodes

    def default_inputs(self) -> Dict[str, DefaultValue]:
        """ Returns all NodeInputs that are bound to a value and not a NodeOutput.

        Returns:
            A dict of DefaultValues.
        """
        return {
            k: v.get_link()
            for k, v in self.inputs().items()
            if isinstance(v.input, DefaultValue)
        }

    def set_draw_node(self, name):
        """ Sets the draw node to a certain node.

        Args:
            name: Name/Id/Key of the draw node.
        """
        self._draw_node = self._nodes[name]
        self._draw_node_name = name

    def draw_node(self):
        """ Returns the Name/Id/Key of the draw node.

        Returns:
            Name of the DrawNode.

        """
        return self._draw_node_name

    def __getitem__(self, item):
        return self._nodes[item]

    def __setitem__(self, key, value):
        self.add_node(value, name=key)

    def __repr__(self):
        return "\n".join(map(lambda x: f"{x[0]}: {x[1]}", self.nodes().items()))

    def to_device(self, device):
        """ Moves data/computation to different devices, typically either 'cuda' or 'cpu'.

        Args:
            device: Name of the device.
        """
        for node in self._nodes.values():
            node.to_device(device)
        self.device = None

    def add_preset(self, key, preset: Preset):
        """ Adds a preset to the graph.

        Args:
            preset_name: The name of the preset to be used as a display name and lookup value.
            preset: The preset object containing all the information about which values should be changed.

        """

        assert isinstance(preset, Preset)
        assert all(k in self.default_inputs().keys() for k in preset.values.keys())
        self.presets[str(key)] = preset

    def set_from_preset(self, key):
        """ Sets the graph NodeInputs to the values specified in the preset that has
        been added under the name 'preset_name'.
        self.presets[str(key)] = preset

        Args:
            preset_name: The lookup key for the preset.

        Returns:
            The set/changed values. These are relevant for  correct display in the
            frontend.
        """

        return self.presets[str(key)].apply(self.nodes())
