=================
Installing Hippo7
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Overview
--------

Hippo7 is written in Python_ and supports Python 3.7+.

.. _Python: https://docs.python-guide.org/

The repo has the following prerequisites:

- You need to have poetry_ installed
- You also might need to install the following packages

.. code-block:: shell

    sudo apt-get install -y ffmpeg libglu1-mesa-dev mesa-common-dev freeglut3-dev


- If you want cuda support for a faster inference of the GAN, you also need to install cuda. Since this is highly dependant on your distribution and GPU, please search the internet for a tutorial for your specific case.


Linux
-----


1. Pull the repository from the `github repository`_.

.. code-block:: shell

    git clone https://gitlab.com/luminovo/hippo7/

2. Install the dependencies with

.. code-block:: shell

    poetry install


3. Start the server and the tool.

.. code-block:: shell

    $ python3 render_server.py

.. code-block:: shell

    $ python3 frontend_app.py

If you now see two windows, the stream and the visual jockey tool, then everything is setup correctly and you can start
configuring it manually.



Windows
-------

We did not test the tool under windows, since python and cuda is not always trivial to setup under this operating system.
But in theory, it should work if you follow roughly along the linux installation steps.



See :doc:`/docs/usage/quickstart` for an in-depth
introduction to configuring the tool.

.. _`github repository`: https://gitlab.com/luminovo/hippo7/
.. _poetry: https://python-poetry.org/
