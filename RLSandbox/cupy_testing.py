#import numpy as np
import cupy as np

seed = np.random.randint(0, 2**16)
print(seed)

np_rng = np.random.default_rng(seed)

print(np_rng)