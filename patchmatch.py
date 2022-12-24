import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import njit, boolean, int64, float64, complex128

np.random.seed(0)

# --------------------------
# numba-compatible functions
# --------------------------

FACTORIALS_LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@njit
def factorial(n):
    """Numba-compatible factorial function."""
    if n > 20:
        raise ValueError
    return FACTORIALS_LOOKUP_TABLE[n]

@njit
def h(x):
    """Bi-cubic interpolation function."""
    if np.abs(x) <= 1:
        return 3 / 2 * np.abs(x)**3 - 5 / 2 * x**2 + 1
    elif np.abs(x) <= 2:
        return - 1 / 2 * np.abs(x)**3 + 5 / 2 * x**2 - 4 * np.abs(x) + 2
    else:
        return 0.

@njit
def double2single_zernike_index(radial_degree, azimuthal_degree):
    """Convert the double indexing (radial_degree, azimuthal degree) of Zernike polytnomials to a single indexing.
    Only polynomials with both positive radial and azimuthal degrees are indexed.
    """
    assert (radial_degree - azimuthal_degree) % 2 == 0
    assert radial_degree > 0 and azimuthal_degree > 0
    n_smaller_polynomials = (radial_degree // 2) * ((radial_degree + 1) // 2)  # number of polynomials with radial degree less than radial_degree
    if radial_degree % 2 == 0:
        return n_smaller_polynomials + azimuthal_degree // 2 - 1
    else:
        return n_smaller_polynomials + (azimuthal_degree - 1) // 2


# ------------------------------------
# numba class attribute specifications
# ------------------------------------

spec = [
    ("im", float64[:, :, :]),
    ("m", int64),
    ("n", int64),
    ("p", int64),
    ("max_zrd", int64),
    ("min_dn", int64),
    ("n_rs_candidates", int64),
    ("n_propagations", int64),
    ("zernike", boolean),
    ("zernike_filters", complex128[:, :, :]),
    ("zernike_moments", float64[:, :, :]),
    ("vect_field", int64[:, :, :]),
    ("dist_field", float64[:, :])
]

# ----------------
# global variables
# ----------------

OFFSETS = np.array([(0, -1), (-1, -1), (-1, 0), (-1, 1)])  # offsets for propagation (in PatchMatch.scan): left, top left, top, top right
N_OFFSETS = len(OFFSETS)

# Coefficients C for the computation of the Zernike filters
# See `Automatic Detection of Internal Copy-Move Forgeries in Images`, Thibaud Ehret, 2018.
MAX_ZERNIKE_ORDER = 10
C = np.zeros((MAX_ZERNIKE_ORDER + 1, MAX_ZERNIKE_ORDER + 1, MAX_ZERNIKE_ORDER // 2 + 1), dtype=np.float64)
for rd in range(1, MAX_ZERNIKE_ORDER + 1):  # radial degree
    for ad in range(1, rd + 1):  # azimuthal degree
        if (rd - ad) % 2 == 0:
            for s in range((rd - ad) // 2 + 1):
                num = (-1)**s * factorial(rd - s)
                denum = (rd - 2 * s + 2) * factorial((rd + ad) // 2 - s) * factorial((rd - ad) // 2 - s)
                C[rd, ad, s] = num / denum

# ----------------
# PatchMatch class
# ----------------

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
    
    max_zrd : int
        maximum radial degree of Zernike polynomials used to compute the Zernike moments
    
    min_dn : int
        lower bound imposed on the infinite norm of displacement vectors (`dn` stands for displacement norm)
    
    n_rs_candidates : int
        number of candidates in the random search phase
        We choose n_rs_candidates new candidates randomely in squares of size 2**i, 0 <= i <= n_rs_candidates - 1.
    
    n_propagations : int
        used to record the number of changes in vect_field during a single scan of PatchMatch
    
    zernike : bool
        Whether to use Zernike moments as features instead of RGB patches.
    
    zernike_filters : array-like, shape (m, n, n_filters)
        array of convolution kernels used to compute the Zernike moments for each patch
    
    zernike_moments : array-like, shape (m, n, 3 * n_filters)
        array of Zernike moments, used as features for the PatchMatch algorithm.
        3 * n_filters channels <=> 1 channel for each Zernike polynomial and for each RGB channel.
    
    vect_field : array-like, shape (m, n, 2)
        displacement field, = one displacement vector for each pixel
        vect_field[i, j, 0] is the i coordinate of the displacement vector
        vect_field[i, j, 1] is the j coordinate of the displacement vector
    
    dist_field : array-like, shape (m,n)
        dist_field[i, j] is the 'distance' between the patch centered at (i, j) and its 'favorite' (see glossary).

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


    def __init__(self, im, p, max_zrd, min_dn, n_rs_candidates, init_method=2, zernike=True):
        """
        Instantiates the PatchMatch algorithm.
        
        Parameters
        ----------
        im, p, max_zrd, min_dn, n_rs_candidates: See class documentation.

        init_method : int
            Method to use to initialize the displacement field.
        """
        self.im = im
        self.m, self.n, _ = im.shape
        self.p = p
        assert min(self.m, self.n) >= 2 * self.p + 1, "At least one full patch must be contained in the image."
        assert self.p >= 2, "p must statisfy p >= 2"  # to avoid index out of range in 1st order propagation in self.scan
        self.max_zrd = max_zrd
        self.min_dn = min_dn
        self.n_rs_candidates = n_rs_candidates
        self.zernike = zernike
        self.n_propagations = 0  # number of change in vect_field for each scan
        if zernike:
            self.create_zernike_filters()
            self.create_zernike_moments()
        if init_method == 1:
            self.create_vect_field1()
        elif init_method == 2:
            self.create_vect_field2()
        else:
            raise ValueError
        self.create_dist_field()
    
    # ----------------------------------------
    # zernike_moments initialization functions
    # ----------------------------------------
    
    def create_zernike_filters(self):
        """Compute filters F^{n, m}_{x, y} as defined in `Automatic Detection of Internal Copy-Move Forgeries in Images`, Thibaud Ehret, 2018."""
        p, max_zrd = self.p, self.max_zrd
        n_filters = double2single_zernike_index(self.max_zrd + 1, self.max_zrd % 2 + 1)
        self.zernike_filters = np.zeros((2 * p + 1, 2 * p + 1, n_filters), dtype=np.complex128)
        # For each pixel in polar coordinates
        for rho in range(p):  # radius
            for theta in range(4 * (2 * rho + 1) - 1):  # azimuthal angle
                # For each Zernike polynomial
                for rd in range(1, max_zrd + 1):  # radial degree
                    for ad in range((rd - 1) % 2 + 1, rd + 1, 2):  # azimuthal degree
                        filter_idx = double2single_zernike_index(rd, ad)  # index of current Zernike filter
                        w = 0
                        # Radial integration
                        for s in range((rd - ad) // 2 + 1):
                            a1 = ((rho + 1) / p)**(rd - 2 * s + 2)
                            a2 = (rho / p)**(rd - 2 * s + 2)
                            w += C[rd, ad, s] * (a1 - a2)
                        # Azimuthal integration
                        dtheta = 2 * np.pi / (4 * (2 * rho + 1))  # elementary angle
                        if ad == 0:  # condition never met in current implementation, but here for future uses.
                            w *= dtheta
                        else:
                            a1 = np.exp(- 1j * ad * (theta + 1) * dtheta)
                            a2 = np.exp(- 1j * ad * theta * dtheta)
                            w *= 1j / ad * (a1 - a2)
                        # Interpolation
                        i0 = rho * np.cos(dtheta * theta)
                        j0 = rho * np.sin(dtheta * theta)
                        imin = int(np.floor(i0) - 1)
                        imax = int(np.floor(i0) + 2)
                        jmin = int(np.floor(j0) - 1)
                        jmax = int(np.floor(j0) + 2)
                        for i in range(imin, min(imax, p) + 1):
                            for j in range(jmin, min(jmax, p) + 1):
                                self.zernike_filters[i + p, j + p, filter_idx] += h(i0 - i) * h(j0 - j) * w

    def create_zernike_moments(self):
        m, n, p = self.m, self.n, self.p
        n_filters = self.zernike_filters.shape[-1]
        self.zernike_moments = np.zeros((m, n, 3 * n_filters), dtype=np.float64)
        for i in range(p, m - p):
            for j in range(p, n - p):
                a = np.zeros(3 * n_filters, dtype=np.complex128)
                for di in range(-p, p + 1):
                    for dj in range(-p, p + 1):
                        coefs = self.zernike_filters[di + p, dj + p]
                        rgb = self.im[i + di, j + dj]
                        a += np.outer(coefs, rgb).flatten()
                self.zernike_moments[i, j] = np.abs(a)
    
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
                if np.abs(end_points[i, j, 0] - i) >= self.min_dn:  # if |di| >= T, sample dj among all admissible values
                    end_points[i, j, 1] = np.random.randint(low=p, high=n - p)
                else:  # else, sample dj among admissible values s.t. |dj| >= T
                    left = max(0, j - self.min_dn - p + 1)  # number of admissible j2 coordinates s.t. j2 < j
                    right = max(0, n - j - self.min_dn - p)  # number of admissible j2 coordinates s.t. j2 > j
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
        to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.min_dn  # kwarg axis for np.max is not supported in numba???
        while np.any(to_small):  # resample the displacement vectors until they match the condition
            for i in range(m):
                for j in range(n):
                    if to_small[i, j]:
                        end_points[i, j, 0] = np.random.randint(low=p, high=m - p)
                        end_points[i, j, 1] = np.random.randint(low=p, high=n - p)
            diff = np.abs(end_points - start_points)
            to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.min_dn  # kwarg axis of np.max is not supported in numba???
        
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

    def patch_features(self, i, j):
        """Return features of patch centered at (i, j)."""
        if self.zernike:
            return self.zernike_moments[i, j].reshape((1, 1, -1))  # to have same nb of dimensions in both cases (required by numba)
        else:
            p = self.p
            return self.im[i - p:i + p + 1, j - p:j + p + 1]
    
    def dist(self, i, j, k, l):
        """Return l2 distance between patch centered at (i, j) and patch centered at (k, l)."""
        return np.sqrt(np.sum((self.patch_features(i, j) - self.patch_features(k, l))**2))

    def dist2candidate(self, i, j, k, l):
        """Evaluate the displacement of (k, l) as a potential displacement for (i, j) and return the associated distance."""
        dk, dl = self.vect_field[k, l]
        return self.dist(i, j, i + dk, j + dl)
    
    def test_min_separation(self, di, dj):
        """Test the condition ||(di, dj)||_infty >= T."""
        return np.abs(di) >= self.min_dn or np.abs(dj) >= self.min_dn
    
    def get_min_displacement_norm(self):
        """Get minimum displacement infinite norm over inner image."""
        m, n, p = self.m, self.n, self.p
        absolute_displacements = np.abs(self.vect_field[p:m - p, p:n - p])
        norms = np.maximum(absolute_displacements[..., 0], absolute_displacements[..., 1])
        return np.min(norms)

    def is_in_inner_image(self, i, j):
        """Return True if point (i, j) is in inner image, and False otherwise."""
        m, n, p = self.m, self.n, self.p  
        return i >= p and i < m - p and j >= p and j < n - p

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
                zo_distances = np.Inf * np.ones(N_OFFSETS, dtype=np.float64)
                for c in range(N_OFFSETS):
                    oi, oj = OFFSETS[c]
                    neighbour = (i + oi, j + oj)
                    di, dj = self.vect_field[neighbour]
                    if self.is_in_inner_image(*neighbour) and self.is_in_inner_image(i + di, j + dj):
                        zo_distances[c] = self.dist(i, j, i + di, j + dj)
                # ---------------------
                # 1st order propagation
                # ---------------------
                fo_distances = np.Inf * np.ones(N_OFFSETS, dtype=np.float64)
                for c in range(N_OFFSETS):
                    oi, oj = OFFSETS[c]
                    neighbour1 = (i + oi, j + oj)
                    neighbour2 = (i + 2 * oi, j + 2 * oj)
                    di, dj = 2 * self.vect_field[neighbour1] - self.vect_field[neighbour2]
                    if self.is_in_inner_image(*neighbour2) and self.is_in_inner_image(i + di, j + dj) and self.test_min_separation(di, dj):
                        fo_distances[c] = self.dist(i, j, i + di, j + dj)
                
                all_distances = np.concatenate((zo_distances, fo_distances))

                # Compute best displacement
                idx = np.argmin(all_distances)
                dmin = all_distances[idx]

                # Propagate best displacement
                if dmin < d0:
                    self.dist_field[i, j] = dmin
                    self.n_propagations += 1
                    oi, oj = OFFSETS[idx % N_OFFSETS]
                    if idx < N_OFFSETS:
                        # 0th order propagation
                        self.vect_field[i, j] = self.vect_field[i + oi, j + oj]
                    else:
                        # 1st order propagation
                        self.vect_field[i, j] = 2 * self.vect_field[i + oi, j + oj] - self.vect_field[i + 2 * oi, j + 2 * oj]

    def random_search(self):
        """Function to make the random search"""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m-p):
            for j in range(p, n-p):
                for k in range(self.n_rs_candidates):
                    di, dj = self.vect_field[i, j]
                    di_ = np.random.randint(max(i + di - 2**k, p) - i, min(i + di + 2**k + 1, m - p) - i)
                    dj_ = np.random.randint(max(j + dj - 2**k, p) - j, min(j + dj + 2**k + 1, n - p) - j)
                    if self.test_min_separation(di_, dj_):
                        d_init = self.dist_field[i, j]
                        d_test = self.dist(i, j, i + di_, j + dj_)
                        if d_test < d_init:
                            self.n_propagations += 1
                            self.vect_field[i, j] = np.array([di_, dj_])
    
    def symmetry(self):
        """Enforce the symmetry of the vect_field map."""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m - p):
            for j in range(p, n - p):
                di, dj = self.vect_field[i, j]
                if self.dist_field[i + di, j + dj] > self.dist_field[i, j]:
                    self.n_propagations += 1
                    self.vect_field[i + di, j + dj] = -self.vect_field[i, j]
                    self.dist_field[i + di, j + dj] = self.dist_field[i, j]
    
    def flip(self):
        """Flip image and vector field."""
        self.im = self.im[::-1, ::-1]
        self.vect_field = -self.vect_field[::-1, ::-1]
        self.dist_field = self.dist_field[::-1, ::-1]

    def iterate(self):
        """Run one iteration of the PatchMatch algorithm."""
        for _ in range(2):
            self.n_propagations = 0
            self.scan()
            self.random_search()
            self.symmetry()
            print(self.n_propagations)
            self.flip()

    def run(self, n_iter):
        """
        Run the PatchMatch algorithm and return the resulting vector field.

        Parameters
        ----------
        N : int
            number of iterations in the PatchMatch algorithm
        """
        for _ in range(n_iter):
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
