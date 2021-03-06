===============
Getting Started
===============

Again start the backend and the frontend in this order with following commands:

.. code-block:: shell

    $ poetry run python3 backend_server.py

If you have successfully finished the installation, you should now have this window:

.. image:: /docs/assets/images/backend.png

Once the backend is running, you can start the frontend.

.. code-block:: shell

    $ poetry run python3 frontend_app.py


The  frontend App Hippo7, looking somehow similar to this should appear on your screen:

.. image:: /docs/assets/images/frontend.png

By default the :ref:`Advanced_Setup` is selected. This means we run the 512 BigGan model and the graph consists
of ton of nodes. So as a result you have a lot of options to configure in the frontend.


Understanding the Frontend
--------------------------

Most of the controls in the frontend should be intuitive. Inevitable some parts are more complex and need
additional information to understand them.

The standard places you can look for more information are:

1.Check the currently used graph and follow the node connections either directly in the
:file:`src/hippo7_app/hippo7_backend/graph_setup.py` or via a autogenerated graph with:

.. function:: src.hippo7_app/hippo7_backend.graph_setup.visualize_graph

2.Often times resetting everything to default, the easiest way via the default preset, and then slowly changing
a single parameter can help achieve a better understanding. Unfortunately in some cases parameters will have no affect, if
other settings are turned off.

3.Try to understand the node in the backend. Thereby first check the inputs and outputs.

4.Read the docs.


Render Server
-------------------------

The :file:`src/backend_server.py` is the start point for the backend. Here you choose the BigGan Model and the
graph architecture. Currently you can choose between 3 example graphs or build your own.
The examples are stored in :file:`src/hippo7_app/hippo7_backend/graph_setup`.

Furthermore by running :file:`src/backend_server.py` you start the render server and consequently the http server.

If you ever want to go for a deep dive into the backend use your debugger and this place as staring point.

Building a Graph
----------------

If you feel gated by the current graph or just want to add or remove a node to implement your own idea,
nothing should stop you. It is fairly simple to write new nodes or manipulate the existing graph.
Check out :doc:`/docs/api/backend/graph_setup`


Connection
----------

If no connection can be established between frontend and backend you have multiple locations where you
can start investigating:

1. Check if the backend :file:`src/backend_server` uses the same ip as the frontend :file:`src/ui/hippo7.ini`
2. If you are not in a localhost environment use the following function.

.. function:: src.hippo7_app/hippo7_backend.server.get_ip

CUDA Support
------------

The tool automatically detects if CUDA is installed and will use it.
Since the computation for BigGan are relatively expensive you will definitely need CUDA for a smooth live video,
otherwise you will struggle to reach enough frames per second.

If the aim is to record a video a GPU is not required, since you can set the FPS to a "fixed time" via:

.. function:: src.hippo7_app/hippo7_backend.pipeline.nodes_util.fixed_time_generator

BigGan
------

`BigGan`_ can be used in 3 different resolutions: 128, 256 and 512. You can select your desired model in the
render_server. The higher the resolution, the higher the required computing power.

.. _BigGan: https://arxiv.org/abs/1809.11096
