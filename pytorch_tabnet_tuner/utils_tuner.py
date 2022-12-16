import numpy as np
import json

from pytorch_tabnet.utils import ComplexEncoder


class ComplexEncoderTuner(ComplexEncoder):

    def default(self, obj):
        """A custom function, from ComplexEncoder, that can be used to serialize numpy arrays.
        The original function, from ComplexEncoder, does not serialize numpy arrays of type int64.
        But, it's need serialize numpy arrays of all types.
        """
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
