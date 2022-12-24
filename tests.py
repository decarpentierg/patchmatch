import unittest
import numpy as np
from PIL import Image

import patchmatch as pm

im = Image.open("data/TP_C02_007_copy.png")
im = np.array(im).astype("double")
gt = Image.open("data/TP_C02_007_gt.png")
gt = np.array(gt) > 0
m, n, _ = im.shape


class TestInitialization(unittest.TestCase):
    """Class to test the initialization"""

    def test_init_methods(self):
        for init_method in range(1, 3):
            p, min_dn, n_rs_candidates = 10, 64, 3
            a = pm.PatchMatch(im, p=p, max_zrd=4, min_dn=min_dn, n_rs_candidates=n_rs_candidates)

            # Assert that the minimum inifinite norm of the displacements is >=T
            self.assertGreaterEqual(a.get_min_displacement_norm(), min_dn)

            start_points = np.zeros((m, n, 2), dtype=np.int64)
            start_points[:, :, 0] = np.arange(m).reshape((m, 1))
            start_points[:, :, 1] = np.arange(n).reshape((1, n))

            end_points = (start_points + a.vect_field)[p:m - p, p:n - p]
            self.assertGreaterEqual(np.min(end_points), p)
            self.assertLess(np.max(end_points[..., 0]), m - p)
            self.assertLess(np.max(end_points[..., 1]), n - p)


if __name__ == '__main__':
    unittest.main()
