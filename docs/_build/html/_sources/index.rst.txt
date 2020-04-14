.. Hippo7 documentation master file, created by
   sphinx-quickstart on Sat Mar 21 14:36:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hippo7
==================================
Hippo7 is a python vjing tool for for creation and manipulation of images to music.
It provides the building blocks necessary to create your own projector based light show.

For a first impression check out our demo video:

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/TIPJwD7XZ6o" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



The tool is build in a modular style, similar to an old synthesizer.
The aim is to enable you to easily build your own computation graph with different nodes, that each compute a small part
of the visualization. By combining node inputs and outputs, different effects can be generated.
The tool is build in a way that with low effort you can add new nodes and add your own custom effects.
Further more the frontend is generated automatically as long as you follow some basic rules.

This tool has been developed in the context of our "Interdisciplinary Project" at Luminovo_ and under supervision
of `TUM Professor Diepold`_ by `Benedikt Wiberg`_ and `Paul Kehnel`_.

.. _Benedikt Wiberg: https://github.com/Qway
.. _Paul Kehnel: https://github.com/PKehnel
.. _TUM Professor Diepold: https://www.professoren.tum.de/diepold-klaus/
.. _Luminovo: https://luminovo.ai

Getting started
---------------
.. toctree::
   :maxdepth: 1

   docs/usage/installation.rst
   docs/usage/quickstart.rst
   docs/usage/examples.rst

Troubleshooting
---------------

If you have questions about how to use it or the configuration options.
First check out the :doc:`/docs/usage/examples` or contact us directly via email.

API documentation
-----------------
Backend:

.. toctree::
   :maxdepth: 2

   docs/api/backend/graph_setup.rst
   docs/api/backend/rendergraph.rst
   docs/api/backend/existing_nodes.rst
   docs/api/backend/opengl.rst


Frontend:

.. toctree::
   :maxdepth: 2
   :glob:

   docs/api/frontend/*



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
