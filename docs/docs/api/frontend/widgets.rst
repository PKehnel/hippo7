=================
Widgets
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console


Generic Widgets
---------------

The generic widgets job is to handle and display frequently recurring values, like the node parameters or
certain buttons.

For example a float will always be displayed with a slider. Additional information like the min/max or the
stepsize can be parsed alongside.

.. autoclass:: src.ui.widgets.custom_widgets.NodeWidget
.. autoclass:: src.ui.widgets.custom_widgets.NodeSwitch
.. autoclass:: src.ui.widgets.custom_widgets.NodeSlider
.. autoclass:: src.ui.widgets.custom_widgets.NodeColor
.. autoclass:: src.ui.widgets.custom_widgets.NodeDisplay
.. autoclass:: src.ui.widgets.custom_widgets.CustomButton
.. autoclass:: src.ui.widgets.custom_widgets.CustomToggleButton

Custom Widgets
--------------

Parts that are only used once or have a low probability that they will be reused, dont have to be written in such
a general style.

Class Selection
^^^^^^^^^^^^^^^

The node class selection handles from which classes we sample.
In the frontend you can manipulate, reorder ... this pool with the class selection widget.

.. _Class_Selection_Widget:

.. automodule:: src.ui.widgets.class_selection
    :members:

.. _Bpm_Widget:

BPM Manager
^^^^^^^^^^^

This widget handles detecting, setting or adjusting of the BPM. This is the only computation actually done in
the frontend, since often the frontend device is physically next to the music.

.. automodule:: src.ui.widgets.bpm_widget
    :members:
    :exclude-members: BPMAnimation

