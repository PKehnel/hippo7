import sys
from copy import copy

import torch
from torch import Tensor as TorchTensor

from hippo7_app.hippo7_backend.pipeline.nodes_util import BeatTimeMessage


class Ntype:
    @classmethod
    def type_from_value(cls, value):
        value_type = type(value)
        resolution_dict = {
            int: Int,
            float: Float,
            bool: Bool,
            str: String,
            TorchTensor: Tensor,
        }
        try:
            return resolution_dict[value_type]()
        except KeyError:
            raise TypeError(f"Can't resolve type of value")

    def default_value(self):
        return None

    def copy(self):
        return copy(self)

    def cast(self, value):
        return value


class Int(Ntype):
    def __init__(self, lower_bound=-sys.maxsize, upper_bound=sys.maxsize):

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def default_value(self):
        return 0


class UInt(Int):
    def __init__(self, lower_bound=0, upper_bound=sys.maxsize):
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound)


class Float(Ntype):
    def __init__(self, lower_bound=sys.float_info.min, upper_bound=sys.float_info.max):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def default_value(self):
        return 0.0


class List(Ntype):
    def __init__(self, value_type):
        self.value_type = value_type

    def default_value(self):
        return []


class ShaderProgram(Ntype):
    pass


class Image(Ntype):
    pass


class Window(Ntype):
    pass


class Bool(Ntype):
    def default_value(self):
        return False


class Tuple(Ntype):
    def __init__(self, *args):
        self.fields = list(args)

    def __len__(self):
        return len(self.fields)

    def default_value(self):
        return ()

    def cast(self, value):
        return list(value)


class FixedSelector(Ntype):
    def __init__(self, ntype, known_values=None):
        self.ntype = ntype
        self.known_values = known_values if known_values else []


class Selector(Ntype):
    def __init__(self, ntype, known_values=None):
        self.ntype = ntype
        self.known_values = known_values if known_values else []

    def copy(self):
        return Selector(self.ntype, copy(self.known_values))


class RGB(Tuple):
    def __init__(self):
        super().__init__(
            (UInt(upper_bound=255), UInt(upper_bound=255), UInt(upper_bound=255))
        )

    def default_value(self):
        return 0, 0, 0


class BeatTime(Ntype):
    def default_value(self):
        return BeatTimeMessage()


class String(Ntype):
    def default_value(self):
        return ""


class Sampler(Ntype):
    pass

    def default_value(self):
        return None


class Tensor(Ntype):
    def __init__(self, shape=None):
        self.shape = shape

    def cast(self, value):
        return torch.as_tensor(value)


class Vector2(Tuple):
    def __init__(
        self,
        lower_bound_x=sys.float_info.min,
        upper_bound_x=sys.float_info.max,
        lower_bound_y=None,
        upper_bound_y=None,
    ):
        lower_bound_y = lower_bound_y if lower_bound_y is not None else lower_bound_x
        upper_bound_y = upper_bound_y if upper_bound_y is not None else upper_bound_x
        super(Vector2, self).__init__(
            Float(lower_bound_x, upper_bound_x), Float(lower_bound_y, upper_bound_y)
        )


class Vector3(Tuple):
    def __init__(
        self, lower_bound=sys.float_info.min, upper_bound=sys.float_info.max,
    ):
        super(Vector3, self).__init__(
            Float(lower_bound, upper_bound),
            Float(lower_bound, upper_bound),
            Float(lower_bound, upper_bound),
        )


class MathFunction(Ntype):
    pass


class Base2Int(Int):
    pass
