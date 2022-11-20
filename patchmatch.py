"""
m : image height
n : image length
p : 2*p+1 is the patch length
im : numpy array of float of size (m,n,3) : the image
vect_field : numpy array of size (m,n,3) : field of displacement vector for each pixel + distance to the matched patch
T : int : inferior limit for the infinite norm of displacements vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int64, float64

np.random.seed(0)

spec = [
    ("im", float64[:, :, :]),
    ("m", int64),
    ("n", int64),
    ("T", int64),
    ("p", int64),
    ("N", int64),
    ("L", int64),
    ("vect_field", int64[:, :, :]),
    ("dist_field", float64[:, :])
]

@jitclass(spec)
class PatchMatch:
    def __init__(self, im, T, p, N, L):
        """
        Instantiate the PatchMatch algorithm 
        
        Parameters
        ----------
        im : array-like, shape (m, n, 3)
            image
        T : int
            inferior limit for the infinite norm of displacements vectors
        p : int
            a patch is a shape (2*p+1,2*p+1)
        N : int
            number of iteration in the PatchMatch algorithm
        L : int
            number of candidate in the random search phase. We choose L new candidates randomely in square of size 2**(i-1) (i \in [1,L])
        """
        self.im = im
        self.m, self.n, _ = im.shape
        self.T = T
        self.p = p
        self.N = N
        self.L = L
        self.create_vect_field()
        self.create_dist_field()


    def create_vect_field(self):
        """
        Create a random vect_field where every vector are biger than T in infinite norm  

        vect_field : array-like, shape (m, n, 3)
            Field of displacement vectors for each pixel. 
            vect_field[i,j,0] is the x coordonate of the displacement vector. 
            vect_field[i,j,1] is the y coordonate of the displacement vector. 
            vect_field[i,j,2] is the distance between the patch and the moved patch.
        """
        m, n, p = self.m, self.n, self.p
        self.vect_field = np.zeros((m, n, 2), dtype=np.int64)

        # generate the target point with the constraint T
        # i coordinate
        I = np.arange(p, m-p)
        for i in range(m):
            idx = np.where(np.abs(I-i) >= self.T)
            self.vect_field[i,:,0] = np.random.choice(I[idx], n)
        # j coordinate
        J = np.arange(p, n-p)
        for j in range(n):
            idx = np.where(np.abs(J-j) >= self.T)
            self.vect_field[:,j,1] = np.random.choice(J[idx], m)

        #compute the displacement vectors
        pos = np.zeros((m, n, 2), dtype=np.int64)
        pos[:, :, 0] = np.arange(m).reshape((m, 1))
        pos[:, :, 1] = np.arange(n).reshape((1, n))
        self.vect_field = self.vect_field - pos


    def create_dist_field(self):
        # Compute the distances between a patch and its matched patch
        m, n, p = self.m, self.n, self.p
        self.dist_field = np.zeros((m, n), dtype=np.float64)
        for i in range(p, m-p):
            for j in range(p, n-p):
                self.dist_field[i,j] = self.dist2candidate(i, j, i, j)


    def patch(self, i, j):
        """Return patch centered at (i, j)."""
        p = self.p
        return self.im[i-p:i+p+1, j-p:j+p+1]
        
    
    def dist(self, i, j, k, l):
        """Return l2 distance between patch centered at (i, j) and patch centered at (k, l)."""
        # print(i, j, k, l)
        return np.sqrt(np.sum((self.patch(i, j) - self.patch(k, l))**2))
    

    def dist2candidate(self, i, j, k, l):
        """Evaluate the displacement of (k, l) as a potential displacement for (i, j) and return the associated distance."""
        dk, dl = self.vect_field[k, l]
        return self.dist(i, j, i+dk, j+dl)
    

    def test_min_separation(self, x, y):
        """Test if the displacement with the vector (x,y) is bigger in infinite norm than T"""
        T = self.T
        return (np.abs(x)>=T and np.abs(y)>=T)

    def scan(self):
        """Run a raster scan over the image and propagate displacement vectors."""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m-p):
            for j in range(p, n-p):
                # Evaluate distance to the current nearest neighboor
                d0 = self.dist_field[i, j]
                # Evaluate distance to the candidate defined by the displacement of the pixel above
                if i > p and i + self.vect_field[i-1, j, 0] + p < m:
                    d_up = self.dist2candidate(i, j, i-1, j)
                else:
                    d_up = np.Inf
                # Evaluate distance to the candidate defined by the displacement of the pixel to the left
                if j > p and j + self.vect_field[i, j-1, 1] + p < n:
                    d_left = self.dist2candidate(i, j, i, j-1)
                else:
                    d_left = np.Inf
                # Compute best displacement
                idx = np.argmin(np.array([d0, d_up, d_left], dtype=np.float64))
                # Propagate best displacement
                if idx == 1:
                    self.vect_field[i, j] = self.vect_field[i-1, j]
                    self.dist_field[i, j] = d_up
                if idx == 2:
                    self.vect_field[i, j] = self.vect_field[i, j-1]
                    self.dist_field[i, j] = d_left
    

    def flip(self):
        """Flip image and vector field."""
        self.im = self.im[::-1, ::-1]
        self.vect_field = -self.vect_field[::-1, ::-1]


    def random_search(self):
        """Function to make the random search"""
        m, n, p = self.m, self.n, self.p
        for i in range(p, m-p):
            for j in range(p, n-p):
                for k in range(self.L):
                    di, dj = self.vect_field[i, j]
                    di_ = np.random.randint(max(i + di - 2**(k - 1), p) - i, min(i + di + 2**(k - 1) + 1, m - p) - i)
                    dj_ = np.random.randint(max(j + dj - 2**(k - 1), p) - j, min(j + dj + 2**(k - 1) + 1, n - p) - j)
                    if self.test_min_separation(di_, dj_):
                        d_init = self.dist_field[i, j]
                        d_test = self.dist(i, j, i + di_, j + dj_)
                        if d_test < d_init:
                            self.vect_field[i, j] = np.array([di_, dj_])


    def run(self):
        """Run the PatchMatch algorithm and return the resulting vector field."""
        for _ in range(self.N):
            self.scan()
            self.flip()
            self.scan()
            self.flip()
            self.random_search()
        return self.vect_field



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
    plt.imshow(pm_.im.astype("uint8"))
    for i in range(0, pm_.m, step):  # for each pixel in the mask
        for j in range(0, pm_.n, step):
            if mask[i, j] > 0:
                plt.arrow(j, i, *pm_.vect_field[i, j, ::-1], **default_kwargs)
    plt.show()
