import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int64, float64
from scipy.special import factorial

np.random.seed(0)

spec = [
    ("im", float64[:, :, :]),
    ("m", int64),
    ("n", int64),
    ("p", int64),
    ("T", int64),
    ("N", int64),
    ("L", int64),
    ("cnt",int64),
    ("vect_field", int64[:, :, :]),
    ("dist_field", float64[:, :])
]


OFFSETS = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]  # offsets for propagation (in PatchMatch.scan): left, top left, top, top right


# @jitclass(spec)
class PatchMatch:
    """
    Class to implement the PatchMatch algorithm.

    Attributes
    ----------
    im : array-like, shape (m, n, 3)
        image
    m : int
        image height
    n : int
        image length
    p : int
        half size of patches, i.e. patches have shape (2p+1, 2p+1, 3)
    vect_field : array-like, shape (m, n, 2)
        displacement field, = one displacement vector for each pixel
        vect_field[i, j, 0] is the i coordinate of the displacement vector
        vect_field[i, j, 1] is the j coordinate of the displacement vector
    dist_field : array-like, shape (m,n)
        dist_field[i, j] is the 'distance' between the patch centered at (i, j) and its 'favorite' (see glossary).
    T : int
        lower bound imposed on the infinite norm of displacement vectors
    N : int
        number of iterations in the PatchMatch algorithm
    L : int
        number of candidates in the random search phase
        We choose L new candidates randomely in squares of size 2**(i-1), 1 <= i <= L.
    cnt : int
        used to record the number of changes in vect_field during a single scan of PatchMatch

    Glossary
    --------
    *   'Inner image': image[p:m - p, p:n - p], i.e. pixels of the image that are the center of a patch included in the image.
    *   A 'displacement vector' (di, dj) maps a 'start point' (i, j) to an 'end point' (i2, j2)=(i + di, j + dj).
            Both the start and the end point must be in the inner image.
    *   'Admissible values' for a displacement vector associated to start point (i, j) are the values that maps it to an end point
            in the inner image.
    *   If a patch P1 is mapped to a patch P2 via the displacement field, P2 is called the 'favorite' of P1.
    *   The 'ground distance' between two patches is the distance between their centers along the image.
    *   The 'distance' between two patches is their distance in the metric space of patches.
    """


    def __init__(self, im, p, T, N, L, init_method=2):
        """
        Instantiates the PatchMatch algorithm.
        
        Parameters
        ----------
        im, p, T, N, L: See class documentation.

        init_method : int
            Method to use to initialize the displacement field.
        """
        self.im = im
        self.m, self.n, _ = im.shape
        self.p = p
        assert min(self.m, self.n) >= 2 * self.p + 1, "At least one full patch must be contained in the image."
        assert self.p >= 2, "p must statisfy p >= 2"  # to avoid index out of range in 1st order propagation in self.scan
        self.T = T
        self.N = N
        self.L = L
        self.cnt = 0  # number of change in vect_field for each scan
        if init_method == 1:
            self.create_vect_field1()
        elif init_method == 2:
            self.create_vect_field2()
        else:
            raise ValueError()
        self.create_dist_field()
    
    # -----------------------------------
    # vect_field initialization functions
    # -----------------------------------
    # Following functions sample a new random displacement field and assign it to self.vect_field. Several methods are available.

    def create_vect_field1(self):
        """
        Assigns a new random displacement field to self.vect_field.
        1st method: 
            For each pixel of the inner image:
            *   Sample the di coordinate of the displacement vector randomly with a uniform distribution among admissible values.
            *   If |di| >= T, sample the dj coordinate randomly with a uniform distribution among all admissible values.
            *   Else, sample the dj coordinate randomly with a uniform distribution among admissible values s.t. |dj| >= T.
        """
        m, n, p = self.m, self.n, self.p
        end_points = np.zeros((m, n, 2), dtype=np.int64)

        # coordinates of start points (=meshgrid)
        start_points = np.zeros((m, n, 2), dtype=np.int64)
        start_points[:, :, 0] = np.arange(m).reshape((m, 1))
        start_points[:, :, 1] = np.arange(n).reshape((1, n))
        end_points[:, :, :] = start_points  # set all displacement vectors to 0 (because vect_field = end_points - start_points)

        # sample i2 coordinates for start points in the inner image
        end_points[p:m - p, p:n - p, 0] = np.random.randint(low=p, high=m - p, size=(m - 2 * p, n - 2 * p))

        # sample j2 coordinates for start points in the inner image
        for i in range(p, m - p):
            for j in range(p, n - p):
                if np.abs(end_points[i, j, 0] - i) >= self.T:  # if |di| >= T, sample dj among all admissible values
                    end_points[i, j, 1] = np.random.randint(low=p, high=n - p)
                else:  # else, sample dj among admissible values s.t. |dj| >= T
                    left = max(0, j - self.T - p + 1)  # number of admissible j2 coordinates s.t. j2 < j
                    right = max(0, n - j - self.T - p)  # number of admissible j2 coordinates s.t. j2 > j
                    alea = np.random.randint(low=0, high=left + right)
                    if alea < left:  # j2 < j
                        end_points[i, j, 1] = p + alea
                    else:  # j2 > j
                        end_points[i, j, 1] = n - p - 1 - (alea - left)
        
        self.vect_field = end_points - start_points  # displacement vectors

    def create_vect_field2(self):
        """
        Assigns a new random displacement field to self.vect_field.
        2nd method: Resample displacement vectors that don't satisfy the condition on the infinite norm until all of them do.
        """
        m, n, p = self.m, self.n, self.p
        end_points = np.zeros((m, n, 2), dtype=np.int64)

        # coordinates of start points (=meshgrid)
        start_points = np.zeros((m, n, 2), dtype=np.int64)
        start_points[:, :, 0] = np.arange(m).reshape((m, 1))
        start_points[:, :, 1] = np.arange(n).reshape((1, n))

        # sample end_points
        end_points[:, :, 0] = np.random.randint(low=p, high=m - p, size=(m, n))
        end_points[:, :, 1] = np.random.randint(low=p, high=n - p, size=(m, n))

        # enforce condition on the infinite norm of the displacement vectors by resampling the vectors that don't satisfy
        # the condition, until all of them do.
        diff = np.abs(end_points - start_points)  # absolute values of displacement vectors coordinates
        to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.T  # kwarg axis for np.max is not supported in numba???
        while np.any(to_small):  # resample the displacement vectors until they match the condition
            for i in range(m):
                for j in range(n):
                    if to_small[i, j]:
                        end_points[i, j, 0] = np.random.randint(low=p, high=m - p)
                        end_points[i, j, 1] = np.random.randint(low=p, high=n - p)
            diff = np.abs(end_points - start_points)
            to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.T  # kwarg axis of np.max is not supported in numba???
        
        self.vect_field = end_points - start_points  # displacement vectors

    # -----------------------------------
    # dist_field initialization functions
    # -----------------------------------

    def create_dist_field(self):
        """Create an array of the distances of the patches to their favorites and assign it to self.dist_field."""
        m, n, p = self.m, self.n, self.p
        self.dist_field = np.zeros((m, n), dtype=np.float64)
        for i in range(p, m - p):
            for j in range(p, n - p):
                self.dist_field[i, j] = self.dist2candidate(i, j, i, j)

    # --------------------
    # patch-wise functions
    # --------------------

    def patch(self, i, j):
        """Return patch centered at (i, j)."""
        p = self.p
        return self.im[i - p:i + p + 1, j - p:j + p + 1]
    
    def dist(self, i, j, k, l):
        """Return l2 distance between patch centered at (i, j) and patch centered at (k, l)."""
        return np.sqrt(np.sum((self.patch(i, j) - self.patch(k, l))**2))

    def dist2candidate(self, i, j, k, l):
        """Evaluate the displacement of (k, l) as a potential displacement for (i, j) and return the associated distance."""
        dk, dl = self.vect_field[k, l]
        return self.dist(i, j, i + dk, j + dl)
    
    def test_min_separation(self, di, dj):
        """Test the condition ||(di, dj)||_infty >= T"""
        return np.abs(di) >= self.T or np.abs(dj) >= self.T

    def is_in_inner_image(self, i, j):
        m, n, p = self.m, self.n, self.p  
        return i >= p and i < m - p and j >= p and j < n - p

    # Zernike moments
    
    def unique_zernike_moment(self, i, j, p, u, v):
        """
        Compute the Zernike moment of order u, v for the patch of size (2*p+1,2*p+1) center in (i,j). Compute base on the paper 
        A. Tahmasbi, F. Saki, and S. B. Shokouhi. Classification of benign and malignant masses based on Zernike moments. 
            Comput. Biol. Med., 41(8):726-735, 2011
        """
        Z = 0
        for x in range(-p, p + 1):
            for y in range(-p, p + 1):
                rho = np.sqrt((2 * (x + i) - (2 * p + 1) + 1)**2 + (2 * (y + j) - (2 * p + 1) + 1)**2) / (2 * p + 1)
                theta = np.arctan((2 * p - 2 * x) / (2 * y - 2 * p))
                R = 0
                for s in range((u- np.abs(v)) // 2 + 1):
                    denom = factorial(s) * factorial((u + np.abs(v)) // 2 - s) * factorial((u - np.abs(v)) // 2 - s)
                    R += (-1)**s * factorial(u - s) * rho**(u - 2 * s) / denom
                Z += (u + 1) * self.im[x + i, y + j] * R * np.exp(-1j * v * theta)
        return Z

    def dist_zernike(self, i, j, k, l):
        """Return l2 distance between zernike moment of patch centered at (i, j) and patch centered at (k, l) and of radius self.p.

        zernike_moments are computed on a circle of radius radius centered around center of mass. 
        Returns a vector of absolute Zernike moments through degree for the image im.
        """
        distance = 0
        for u in range(self.p):
            for v in range(-u,u+1,2):
                Z_1 = self.unique_zernike_moment(self, i, j, self.p, u, v)
                Z_2 = self.unique_zernike_moment(self, k, l, self.p, u, v)
                distance += (Z_1-Z_2)**2
        return np.sqrt(distance)

    # --------------------
    # PatchMatch algorithm
    # --------------------

    def scan(self):
        """Run a raster scan over the image and propagate displacement vectors."""
        print("Scan")
        m, n, p = self.m, self.n, self.p
        for i in range(p, m-p):
            for j in range(p, n-p):
                # Evaluate distance to the current nearest neighboor
                d0 = self.dist_field[i, j]
                # ---------------------
                # 0th order propagation
                # ---------------------
                # Zero-th order candidates and associated distances
                zo_distances = [np.Inf for _ in OFFSETS]
                for c in range(len(OFFSETS)):
                    oi, oj = OFFSETS[c]
                    neighbour = (i + oi, j + oj)
                    di, dj = self.vect_field[neighbour]
                    if self.is_in_inner_image(*neighbour) and self.is_in_inner_image(i + di, j + dj):
                        zo_distances[c] = self.dist(i, j, i + di, j + dj)
                # ---------------------
                # 1st order propagation
                # ---------------------
                fo_distances = [np.Inf for _ in OFFSETS]
                for c in range(len(OFFSETS)):
                    oi, oj = OFFSETS[c]
                    neighbour1 = (i + oi, j + oj)
                    neighbour2 = (i + 2 * oi, j + 2 * oj)
                    di, dj = 2 * self.vect_field[neighbour1] - self.vect_field[neighbour2]
                    if self.is_in_inner_image(*neighbour2) and self.is_in_inner_image(i + di, j + dj):
                        fo_distances[c] = self.dist(i, j, i + di, j + dj)
                
                all_distances = np.concatenate((zo_distances, fo_distances))

                # Compute best displacement
                idx = np.argmin(all_distances)
                dmin = all_distances[idx]

                # Propagate best displacement
                if dmin < d0:
                    self.dist_field[i, j] = dmin
                    self.cnt += 1
                    oi, oj = OFFSETS[idx % len(OFFSETS)]
                    if idx < len(OFFSETS):
                        # 0th order propagation
                        self.vect_field[i, j] = self.vect_field[i + oi, j + oj]
                    else:
                        # 1st order propagation
                        self.vect_field[i, j] = 2 * self.vect_field[i + oi, j + oj] - self.vect_field[i + 2 * oi, j + 2 * oj]


    def flip(self):
        """Flip image and vector field."""
        self.im = self.im[::-1, ::-1]
        self.vect_field = -self.vect_field[::-1, ::-1]
        self.dist_field = self.dist_field[::-1, ::-1]

    def random_search(self):
        """Function to make the random search"""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m-p):
            for j in range(p, n-p):
                for k in range(self.L):
                    di, dj = self.vect_field[i, j]
                    di_ = np.random.randint(max(i + di - 2**k, p) - i, min(i + di + 2**k + 1, m - p) - i)
                    dj_ = np.random.randint(max(j + dj - 2**k, p) - j, min(j + dj + 2**k + 1, n - p) - j)
                    if self.test_min_separation(di_, dj_):
                        d_init = self.dist_field[i, j]
                        d_test = self.dist(i, j, i + di_, j + dj_)
                        if d_test < d_init:
                            self.cnt += 1
                            self.vect_field[i, j] = np.array([di_, dj_])
    
    def symmetry(self):
        """Assure the symmetry of the vect_field map"""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m - p):
            for j in range(p, n - p):
                di, dj = self.vect_field[i, j]
                if self.dist_field[i + di, j + dj] > self.dist_field[i, j]:
                    self.vect_field[i + di, j + dj] = -self.vect_field[i, j]
                    self.dist_field[i + di, j + dj] = self.dist_field[i, j]

    def iterate(self):
        for _ in range(2):
            self.cnt = 0
            self.scan()
            self.random_search()
            self.symmetry()
            print(self.cnt)
            self.flip()

    def run(self):
        """Run the PatchMatch algorithm and return the resulting vector field."""
        for _ in range(self.N):
            self.iterate()


def plot_vect_field(pm_, mask, step=100, **kwargs):
    """
    Plot vect_field as arrows above the image
    
    Parameters
    ----------
    pm_ : instance of PatchMatch
    mask : array-like, shape (m, n)
        only vectors whose "roots" are on pixels for wich mask is non-zero will be plotted
    step : int
        step between two plotted vectors
    **kwargs : keyword arguments
        keyword arguments to be passed to plt.arrow (e.g. head_width, head_length, ...)
    """
    default_kwargs = {"width": 1e-3, "head_width": 1, "head_length": 1.5, "length_includes_head": True}
    default_kwargs.update(kwargs)
    default_kwargs["head_length"] = 1.5 * default_kwargs["head_width"]
    plt.imshow(pm_.im.astype("uint8"))
    for i in range(0, pm_.m, step):  # for each pixel in the mask
        for j in range(0, pm_.n, step):
            if mask[i, j] > 0:
                plt.arrow(j, i, *pm_.vect_field[i, j, ::-1], **default_kwargs)
    plt.show()
