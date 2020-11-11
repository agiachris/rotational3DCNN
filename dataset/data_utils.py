import numpy as np


def tensor_from_file(filename):
    """Load Signed Distance Field (SDF) inputs and Distance Field (DF) targets
    from custom format provided by http://graphics.stanford.edu/projects/cnncomplete/
    """
    # load data
    data = np.fromfile(filename, dtype=np.float32)
    # extract header, encoded in uint64
    header = data[:6]
    header = header.tobytes()
    header = np.frombuffer(header, dtype=np.uint64)
    # reshape data into dimensions provided in header
    return data[6:].reshape(header)
