=================
Open GL
=================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

This section covers everything related to the parts of `opengl`_ that are used to render 3D graphics. If you want to
learn more on how to display 3D graphics, head over to their website.
Since python support of 3D graphics is a bit wonky, we use `glumpy`_ as a wrapper for all data transfer between CPU and
GPU. This allows us to use `numpy`_ arrays with a simple syntax to assign new meshes and textures to the GPU.


.. _opengl: https://www.opengl.org/
.. _glumpy: https://glumpy.readthedocs.io/
.. _numpy: https://numpy.org/

Geometry
--------

The current nodes use always the .ply format for geometry. While some correctness checks are used, the current best
working .ply header is as follows::

    ply
    format ascii 1.0
    comment VCGLIB generated
    element vertex 37
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float texture_u
    property float texture_v
    element face 36
    property list uchar int vertex_indices
    property list uchar float texcoord
    end_header


You can find all currently used meshes in ``assets/meshes``.



.. autofunction:: src.hippo7_app.hippo7_backend.opengl.geometry.load_file

Shader
------

Since shaders are hard to switch at runtime, we currently only use one single fragment and vertex shader with numerous
options. While the vertex shader is quite self explanatory, the fragment shader uses numerous flags for effects.

apply_sobel:
    Enables the sobel operator pass to render only edges. The rasterization size can be controlled with ``u_resolution``.

mirror_x, mirror_y:
    Enables mirroring of every texture along the x or y axis(often also called u and v, because its about the texture coordinates).

is_crosshatch:
    Enables the crosshatch filter for a crosshatching effect.

is_inverted:
    Enables the inversion of all color values.

halftone_resolution:
    Enables a halftone effect, if set to a value >0.

repeat:
    Enables a repeating of the texture along the x/u or y/v axis. Note that the value is a Vector of two values and not a Scalar.

texture_scale:
    Scales the texture along along the x/u or y/v axis. Note that the value is a Vector of two values and not a Scalar.
    Can lead to clipping and artifacts, which can be either seen as an artistic effect or simply as ugly.

brightness:
    Multiplies the output colors by this values to modify the output brightness.


Rendering
---------

Currently the rendering process is split over multiple nodes: Shader, GeometryController, DrawNode. It is basically exactly
the same rendering processs as specified in the glumpy docs, except its distributions over multiple nodes to fit the RenderGraph
concept better.

