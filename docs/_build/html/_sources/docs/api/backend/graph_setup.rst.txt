===================
Building your graph
===================

While it is advisable to first try to understand the examples, nothing is stopping you from creating your own graph from
scratch. For this, follow these steps:

First, we need to create our :ref:`RenderGraph<RenderGraph>` object. We then add some default nodes(typically a BPMClock and
a Window node that provides access to the output window).

.. code-block:: python

    g = RenderGraph()
    g.add_node(N.Window(), name="window")
    g.add_node(
        N.BpmClock(bpm=bpm), name="bpm_clock",
    )

Now we can add all our Nodes. For this example, we just want a blinking rectangle displayed.

.. code-block:: python

    # Add two textures
    g["white"] = N.ColorToImage(color=(1,1,1))
    g["black"] = N.ColorToImage(color=(0,0,0))

    # Interpolate between the two textures in the local_time(Time between beats).
    g["output_color"] = N.Interpolate(
        interpolation_value=g["bpm_clock"].local_time,
        input_vector0=g["white"].image,
        input_vector1=g["black"].image,
    )

    # These nodes are needed for the rendering process

    # The shader node is simply a manager for the OpenGL shaders
    g["shader"] = N.Shader()

    # The GeometryController manages the 3D geometry - in our case a simple quad
    g["geo"] = N.GeometryController(
        shader=g["shader"].shader,
        texture= g["output_color"].result,
        mesh="quad",
        window_size=g["window"].size,
    )

    # The DrawNode then renders to the output window
    g["draw"] = N.DrawNode(
        faces=g["geo"].faces,
        shader=g["shader"].shader,
        window=g["window"].window
    )

As a final step, we need to assign which node is the draw node - The node that triggers the update of the output window.

.. code-block:: python

    g.set_draw_node("draw")

As a final step, wrap everything in a function and return our graph `g`:

.. code-block:: python

    def custom_graph():
        g = RenderGraph()
        g["window"] = N.Window()
        g["bpm_clock"] = N.BpmClock(bpm=128)
        g["white"] = N.ColorToImage(color=(1,1,1))
        g["black"] = N.ColorToImage(color=(0,0,0))
        g["output_color"] = N.Interpolate(
            interpolation_value=g["bpm_clock"].local_time,
            input_vector0=g["white"].image,
            input_vector1=g["black"].image,
        )
        g["shader"] = N.Shader()
        g["geo"] = N.GeometryController(
            shader=g["shader"].shader,
            texture= g["output_color"].result,
            mesh="quad",
            window_size=g["window"].size,
        )
        g["draw"] = N.DrawNode(
            faces=g["geo"].faces,
            shader=g["shader"].shader,
            window=g["window"].window
        )
        g.set_draw_node("draw")
        return g

This function can then be inserted in the ``render_server.py`` to use. The ``render_server.py`` manages the creation of the
window and server. Congratulations! If you run the ``render_server.py`` you should get a blinking square as an output.

.. image:: /docs/assets/images/debug_graph.gif

Note that the .gif is not a perfect loop.

Our graph looks as follows(plotted with :py:func:`src.hippo7_app.hippo7_backend.graph_setup.visualize_graph`:



.. graphviz:: ../../assets/graphviz/graphviz_graph.dot



We can now add numerous more effects: For example, we could change the mesh(simply change the line ``mesh="quad",``,
add a rotation(add a :py:class:`src.hippo7_app.hippo7_backend.nodes.ModelMatrix`) or change the interpolation speed/function by adding a
:py:class:`src.hippo7_app.hippo7_backend.nodes.ComplexFunction`.


