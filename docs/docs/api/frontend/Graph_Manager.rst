=================
Graph Manager
=================
.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Graph Manager
-------------


.. autoclass:: src.hippo7_app.hippo7_frontend.graph_manager.GraphManager
    :members:

Observer Pattern
----------------

The `observer pattern`_ is a design pattern where a object, maintains a list of its dependents, called observers, and notifies them automatically of any state changes, usually by calling one of their methods.

In this case the values from  different nodes in the backend are the subjects and the associated widgets are the observers.

..  _observer pattern: https://en.wikipedia.org/wiki/Observer_pattern

.. autoclass:: src.hippo7_app.hippo7_frontend.graph_manager.Observable
    :members:

Backend Communication
---------------------

In the following class the communication with the backend takes place. The interface mirrors the backend API and provides basic
functionality such as sending a updated node and handles first step en- and decoding.


.. autoclass:: src.hippo7_app.hippo7_frontend.backend_interface.BackendInterface
    :members:

.. autofunction:: src.hippo7_app.hippo7_frontend.backend_interface.json_to_graph