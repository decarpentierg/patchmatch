import numpy as np
from PIL import Image

import patchmatch as pm

im = Image.open("data/TP_C01_011_copy_ln20.png")
im = np.array(im).astype("double")[400:]
gt = Image.open("data/TP_C01_011_gt_ln20.png")
gt = np.array(gt)[400:] > 0

a = pm.PatchMatch(im, p=10, T=100, N=1, L=3)

a.iterate()
