import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
import os
import pickle

import patchmatch as pm

DATA = "data/CMFDdb_grip/forged_images/"  # directory where to find forged images
RESULTS = "results/"  # directory where to find the results

ls = sorted([x[:10] for x in os.listdir(DATA) if "copy" in x])

zernike_times = {}
patchmatch_times = {}

for im_name in tqdm(ls):
    # Load image
    im = Image.open(f"{DATA}/{im_name}_copy.png")
    im = np.array(im).astype("double")

    # Initialize PatchMatch
    t0 = time()
    a = pm.PatchMatch(
        im,  # image
        p=10,  # patch half-size
        max_zrd=6,  # maximum Zernike degree
        min_dn=64,   # minimum displacement norm (previously T)
        n_rs_candidates=5,   # number of candidates in the random search phase (previously L)
        init_method=2,  # whether to use create_vect_field1 or create_vect_field2
        zernike=True  # whether to use Zernike moments
    )
    t1 = time()
    zernike_times[im_name] = t1 - t0

    # Run PatchMatch
    t0 = time()
    a.run(10)
    t1 = time()
    patchmatch_times[im_name] = t1 - t0

    # Save results
    res = {attribute[0]:a.__getattribute__(attribute[0]) for attribute in pm.spec[1:]}
    np.savez_compressed(f"{im_name}_results.npz", **res)

with open("zernike_times.pkl", "wb") as file:
    pickle.dump(zernike_times, file)

with open("patchmatch_times.pkl", "wb") as file:
    pickle.dump(patchmatch_times, file)
