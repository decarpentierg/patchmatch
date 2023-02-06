import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import pickle
import argparse

import patchmatch as pm
import postprocessing as pp

parser = argparse.ArgumentParser(description="Run PatchMatch on test database")
parser.add_argument("indices", type=str, help="Run on images TP_C%1_%2 to TP_C%1_%3 where argument=%1,%2,%3")
args = parser.parse_args()
c_idx, start, stop = [int(x) for x in args.indices.split(",")]

DATA = "data/CMFD_DB/"  # directory where to find forged images
RESULTS = "results5/"  # directory where to save the results

ls = sorted([x[:10] for x in os.listdir(DATA) if "copy" in x])

zernike_times = {}
patchmatch_times = {}

def process(idx):
    # Compute image name
    im_name = f"TP_C{c_idx:02d}_{idx:03d}"
    if im_name not in ls:  # if image does not exist, simply skip it
        return

    # Load image and ground truth
    im = Image.open(f"{DATA}/{im_name}_copy.png")
    im = np.array(im).astype("double")
    gt = Image.open(f"{DATA}/{im_name}_gt.png")
    gt = np.array(gt) > 0

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
    niter = 20
    fscores1 = np.zeros(niter)
    fscores2 = np.zeros(niter)
    for i in range(niter):
        a.iterate()
        mask1 = pp.compute_mask_1(a.vect_field, a.m, a.n, a.p)
        mask2 = pp.compute_mask_2(a.vect_field, a.m, a.n, a.p)
        fscores1[i] = pp.fscore(mask1, gt)
        fscores2[i] = pp.fscore(mask2, gt)
    t1 = time()
    patchmatch_times[im_name] = t1 - t0

    # Save results
    attributes = [x[0] for x in pm.spec]
    # res = {attribute:a.__getattribute__(attribute) for attribute in attributes if not attribute in ["im", "zernike_filters", "zernike_moments"]}
    res = {attribute:a.__getattribute__(attribute) for attribute in ["n_performed_iterations", "n_propagations", "sum_of_distances"]}
    res.update({"fscores1": fscores1, "fscores2": fscores2})
    np.savez_compressed(f"{RESULTS}/{im_name}_results.npz", **res)

process_map(process, range(start, stop))

with open(f"{RESULTS}/zernike_times_C{c_idx:02d}_{start:03d}_to_{stop:03d}.pkl", "wb") as file:
    pickle.dump(zernike_times, file)

with open(f"{RESULTS}/patchmatch_times_C{c_idx:02d}_{start:03d}_to_{stop:03d}.pkl", "wb") as file:
    pickle.dump(patchmatch_times, file)
