====================
The Hippo7 App
====================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console


Kivy
----

`Kivy`_ is a free and open source Python library for developing mobile apps and other multitouch application software with a natural user interface.

.. _Kivy: https://kivy.org/#home

The Frontend is build completely with Kivy.

The main page
-------------

The main page consist of a bpm widget in the top left, next to it a overview for all nodes in the backend.
Beneath that the currently selected node is displayed and presets are shown at the bottom.

Furthermore the main page automatically generates the node pages at runtime according to the backend graph.
See next paragraph `Node Pages`_.

.. autoclass:: src.ui.ui_kivy.Hippo7Layout

.. _Node Pages:

Node pages
----------

The node pages  are auto generated at runtime. This works as long as the nodes
follow the :ref:`type conventions <Types>` specified in the backend. Currently supported parameter types include:

String, Int, Double, Bool, Base2Int, Tupple(x,y), RGB

For more specific nodes in the backend a custom widget has to be build, for example the :ref:`Class Selection <Class_Selection_Widget>`.

.. automodule:: src.ui.nodepage
    :members:

Beat Detection and BPM Manager
------------------------------

The beat detection works in realtime and is based on librosa_.

.. _librosa: https://librosa.github.io/librosa/

.. autoclass:: src.ui.util.BeatDetector
    :members:

In the app you can either configure the BPM manually, put it on autodetect or change it via tap detection
using the space bar.

This is implemented in the :ref:`BPM Widget <Bpm_Widget>`.



