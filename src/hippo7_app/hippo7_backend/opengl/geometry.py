from typing import Tuple

import numpy as np
from glumpy import glm
from glumpy import gloo
from plyfile import PlyData
from torch.nn.functional import pad

# The used vertex type def for a numpy array

vtype = [
    ("position", np.float32, 3),
    ("normal", np.float32, 3),
    ("texcoord", np.float32, 2),
]

vertex_type_in = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("texture_u", "<f4"),
        ("texture_v", "<f4"),
    ]
)

face_type_in = np.dtype([("vertex_indices", "O"), ("texcoord", "O")])


def load_file(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a .ply file. Returns all vertex positions, normals and
    texture coordinates and the face definitions.

    Args:
        file (str): A file path.

    Returns:
        A tuple of vertices and faces.
    """

    plyfile = PlyData.read(file)

    assert plyfile.elements[0].data.dtype == vertex_type_in
    # assert plyfile.elements[1].data.dtype < face_type_in

    vertices = plyfile.elements[0].data.view(vtype)
    faces = np.concatenate(plyfile.elements[1].data["vertex_indices"]).view(np.uint32)

    vertices = vertices.view(gloo.VertexBuffer)
    faces = faces.view(gloo.IndexBuffer)
    return vertices, faces


def to_tex_format(state):
    img = (state + 1) / 2
    tensor = img[0].transpose(0, 2).transpose(0, 1).data
    tensor = pad(tensor, (0, 1), "constant", 1)  # add the alpha channel
    tensor = (255 * tensor).byte().contiguous()  # convert to ByteTensor
    return tensor.cpu()


def correctly_rotated_model_matrix():
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, 180, 0, 0, 1)
    return model
