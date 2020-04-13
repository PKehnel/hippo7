=================
The RenderGraph
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

The RenderGraph_ is the *heart of hippo7*. It works similar to a `flow based programming`_ approach. Basically, you first need
to construct a graph that declares the flow of data between numerous nodes that each compute some part of the renderer. Afterwards,
the graph is run for every frame, where every node is run exactly once. While the actual computation is run backwards from the last node,
requesting computation of all needed inputs on the fly, it is easier to think of every node by itself and just assume that all inputs
are already correct for this time step/frame.

.. _flow based programming: https://en.wikipedia.org/wiki/Flow-based_programming

Graph
-----

The Graph itself is a wrapping data structure that allows at runtime manipulation of some of its values.

.. _RenderGraph:
.. autoclass:: src.renderer.pipeline.graph.RenderGraph
    :members:

Presets
^^^^^^^

An important concept for the actual performance using a graph are Presets. Presets can be added to a Graph and allow the
quick change if multiple node inputs(so called DefaultValues_).


.. _Preset:
.. autoclass:: src.renderer.pipeline.graph.Preset
    :members:


.. _TogglePreset:
.. autoclass:: src.renderer.pipeline.graph.TogglePreset
    :members:

Nodes
-----

Nodes are simple computation units in the RenderGraph_ and are all implementations of the same node class.


.. _Nodes:
.. autoclass:: src.renderer.pipeline.graph.Node
   :members:

.. _New_Nodes:

Creating New Nodes
^^^^^^^^^^^^^^^^^^

To create a new custom node, simply follow the this template::

    from renderer.pipeline.graph import Node, NodeInput, NodeOutput
    import renderer.pipeline.types as T

    class ComplexFunction(Node):
        firstInput = NodeInput(1.0, ntype=T.Float(-100, 100))
        secondInput = NodeInput(1.0)

        output = NodeOutput(ntype=T.Float())

        def update(self):
            output = firstInput + secondInput


Basically, you first define the inputs and outputs as class attributes. Note that you can specify the type of the input,
but for some simple types its automatically inferred from the value itself.
Then, you define a custom update method that calculates the output from the inputs.
Afterwards, when creating the graph itself, the values specified in the inputs are converted to DefaultValues_ if not set
to the NodeOutput_ of another Node. For more complex examples it is advisable to look into already existing nodes.

Inputs and Outputs
^^^^^^^^^^^^^^^^^^^

Every node is made of inputs and outputs, that are connected to other Inputs/Outputs or are DefaultValues_.

.. _NodeOutput:
.. autoclass:: src.renderer.pipeline.graph.NodeOutput
    :members:

.. _NodeInput:
.. autoclass:: src.renderer.pipeline.graph.NodeInput
    :members:

.. _DefaultValues:
.. autoclass:: src.renderer.pipeline.graph.DefaultValue
    :members:


.. _Types:

Types
-----

The types are currently for two things: The correct generation of the json serialization and the automatic inferring of
the correct frontend widget to manipulate the value in case of a DefaultValues_. Currently, the following types are available:

.. automodule:: src.renderer.pipeline.types
    :members:
    :undoc-members:
    :exclude-members: cast, copy, default_value, type_from_value

