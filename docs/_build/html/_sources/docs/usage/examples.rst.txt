=================
Examples
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console



Basic setup
-----------

The simplest and minimal setup you can run out of the box is described in the:

.. function:: src.hippo7_app/hippo7_backend.graph_setup.simple_gan_setup

.. image:: /docs/assets/images/simple_gan.gif

If you run the server using this for your graph generation, it will give you a graph looking like this:

.. graphviz:: simple_graph.dot


**Hint:** You can always generate a graph visualisation like above by calling

.. function:: src.hippo7_app/hippo7_backend.graph_setup.visualize_graph



.. _Advanced_Setup:

Advanced setup
--------------
A more advanced configuration can be used with:

.. function:: src.hippo7_app/hippo7_backend.graph_setup.no_gan_setup

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/TIPJwD7XZ6o" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>




This is actually the setup we also used at a live performance so dont feel bad, if you are overwhelmed in the beginning.

Here you have a almost endless variety of options. Some parts like the functions are a not directly understandable,
this is a direct trade off for the automatic generated frontend.

Alternative input
-----------------

There is nothing enforcing you to use BigGan as image input.
You can also run the setup without any visuals, as shown in the following example:

.. function:: src.hippo7_app/hippo7_backend.graph_setup.no_gan_setup

.. image:: /docs/assets/images/nogan.gif

You could also always build your own graph(as described in :doc:`/docs/api/backend/graph_setup`) with your own custom
nodes as input(as seen in :ref:`New_Nodes`).
