import numpy as np
from PIL import Image

import patchmatch as pm

im = Image.open("data/TP_C02_007_copy.png")
im = np.array(im).astype("double")
gt = Image.open("data/TP_C02_007_gt.png")
gt = np.array(gt) > 0

a = pm.PatchMatch(im, p=10, max_zrd=4, min_dn=100, n_rs_candidates=3)

a.iterate()
