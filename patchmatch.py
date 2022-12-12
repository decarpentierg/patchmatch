"""
m : image height
n : image length
p : 2*p+1 is the patch length
im : numpy array of float of size (m,n,3) : the image
vect_field : numpy array of size (m,n,2) : field of displacement vector for each pixel
dist_field : numpy array of size (m,n,1) : distance between a patch (center in this pixel) and the patch obtain after the displacement encoded in vect_field
T : int : inferior limit for the infinite norm of displacements vectors
N : int : number of iteration in patchmatch algorithm
cnt : int : number of change in vect_field for one scan
"""

import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int64, float64, types
from tqdm import tqdm
from mahotas.features import zernike_moments
from scipy.special import factorial

np.random.seed(0)

spec = [
    ("im", float64[:, :, :]),
    ("m", int64),
    ("n", int64),
    ("T", int64),
    ("p", int64),
    ("N", int64),
    ("L", int64),
    ("cnt",int64),
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
        self.cnt = 0  # number of change in vect_field for each scan
        self.create_vect_field()
        self.create_dist_field()

    def create_vect_field(self):
        """
        Create a random vect_field where every vector are biger than T in infinite norm  

        vect_field : array-like, shape (m, n, 2)
            Field of displacement vectors for each pixel. 
            vect_field[i,j,0] is the x coordonate of the displacement vector. 
            vect_field[i,j,1] is the y coordonate of the displacement vector. 
        """
        m, n, p = self.m, self.n, self.p
        self.vect_field = np.zeros((m, n, 2), dtype=np.int64)

        # generate the target point with the constraint T
        # x coordinate
        self.vect_field[:,:,0] = np.random.randint(p+1,m-p,size=(m,n), dtype = np.int64) - np.arange(m, dtype = np.int64)[:,None]

        # y coordinate
        for i in range(m):
            for j in range(n):
                if np.abs(self.vect_field[i,j,0]) > self.T:
                    self.vect_field[i,j,1] = np.random.randint(p+1,n-p)
                else:
                    #research in constant time of an j which assure that the vector is bigger in infinite norm than self.T
                    alea = np.random.randint(p+1,n-p-2*self.T-1)
                    if j-self.T > p and alea < j-self.T:
                        self.vect_field[i,j,1] = alea-j
                    else:
                        self.vect_field[i,j,1] = alea+2*self.T+1-j

    def create_vect_field2(self):
        """
        Create a random vect_field where every vector are biger than T in infinite norm  

        vect_field : array-like, shape (m, n, 2)
            Field of displacement vectors for each pixel. 
            vect_field[i,j,0] is the x coordonate of the displacement vector. 
            vect_field[i,j,1] is the y coordonate of the displacement vector. 
        """
        m, n, p = self.m, self.n, self.p

        # compute the displacement vectors
        pos = np.zeros((m, n, 2), dtype=np.int64)
        pos[:, :, 0] = np.arange(m).reshape((m, 1))
        pos[:, :, 1] = np.arange(n).reshape((1, n))

        self.vect_field = np.zeros((m, n, 2), dtype=np.int64)
        self.vect_field[..., 0] = np.random.randint(low=p, high=m-p, size=(m, n))
        self.vect_field[..., 1] = np.random.randint(low=p, high=n-p, size=(m, n))

        diff = np.abs(self.vect_field - pos)
        to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.T  # kwarg axis for np.max is not supported in numba???
        while np.any(to_small):  # resample the displacement vectors until they match the condition
            for i in range(m):
                for j in range(n):
                    if to_small[i, j]:
                        self.vect_field[i, j, 0] =  np.random.randint(low=p, high=m-p)
                        self.vect_field[i, j, 1] =  np.random.randint(low=p, high=n-p)
            diff = np.abs(self.vect_field - pos)
            to_small = np.maximum(diff[..., 0], diff[..., 1]) < self.T  # kwarg axis for np.max is not supported in numba???
        
        self.vect_field = self.vect_field - pos


    def create_dist_field(self):
        """
        Compute the distances between a patch and its matched patch with the current displacement vector in self.vect_field
        """
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
    

    def unique_zernike_moment(self, i, j, p, u, v):
        """
        Compute the Zernike moment of order u, v for the patch of size (2*p+1,2*p+1) center in (i,j). Compute base on the paper 
        A. Tahmasbi, F. Saki, and S. B. Shokouhi. Classification of benign and malignant masses based on Zernike moments. Comput. Biol. Med., 41(8):726–735, 2011
        """
        Z = 0
        for x in range(-p,p+1):
            for y in range(-p,p+1):
                rho = np.sqrt((2*(x+i)-(2*p+1)+1)**2+(2*(y+j)-(2*p+1)+1)**2)/(2*p+1)
                theta = np.arctan((2*p-2*x)/(2*y-2*p))
                R = 0
                for s in range((u-np.abs(v))//2+1):
                    R += (-1)**s * factorial(u-s) * rho**(u-2*s) /(factorial(s)*factorial((u+np.abs(v))//2-s)*factorial((u-np.abs(v))//2-s))
                Z += (u+1)*self.im[x+i,y+j]*R*np.exp(-1j * v * theta)
        return Z


    def dist_zernike(self, i, j, k, l):
        """Return l2 distance between zernike moment of patch centered at (i, j) and patch centered at (k, l) and of radius self.p.

        zernike_moments are computed on a circle of radius radius centered around center of mass. 
        Returns a vector of absolute Zernike moments through degree for the image im.
        """
        distance = 0
        for u in range(self.p):
            for v in range(-u,u+1,2):
                Z_1 = unique_zernike_moment(self, i, j, self.p, u, v)
                Z_2 = unique_zernike_moment(self, k, l, self.p, u, v)
                distance += (Z_1-Z_2)**2
        return np.sqrt(distance)

    def dist2candidate(self, i, j, k, l):
        """Evaluate the displacement of (k, l) as a potential displacement for (i, j) and return the associated distance."""
        dk, dl = self.vect_field[k, l]
        return self.dist(i, j, i+dk, j+dl)
    

    def test_min_separation(self, x, y):
        """Test if the displacement with the vector (x,y) is bigger in infinite norm than T"""
        T = self.T
        return (np.abs(x)>=T or np.abs(y)>=T)


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
                    self.cnt +=1
                if idx == 2:
                    self.vect_field[i, j] = self.vect_field[i, j-1]
                    self.dist_field[i, j] = d_left
                    self.cnt +=1
        
    

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
                    di_ = np.random.randint(max(i + di - 2**(k - 1), p) - i, min(i + di + 2**(k - 1) + 1, m - p) - i)
                    dj_ = np.random.randint(max(j + dj - 2**(k - 1), p) - j, min(j + dj + 2**(k - 1) + 1, n - p) - j)
                    if self.test_min_separation(di_, dj_):
                        d_init = self.dist_field[i, j]
                        d_test = self.dist(i, j, i + di_, j + dj_)
                        if d_test < d_init:
                            self.cnt += 1
                            self.vect_field[i, j] = np.array([di_, dj_])
    

    def symetry(self):
        """Assure the symetry of the vect_field map"""
        m, n = self.m, self.n
        for i in range(m):
            for j in range(n):
                di, dj = self.vect_field[i, j]
                if self.dist_field[i+di, j+dj] > self.dist_field[i,j]:
                    self.vect_field[i+di,j+dj] = -self.vect_field[i,j]
                    self.dist_field[i+di,j+dj] = self.dist_field[i,j]

    def iterate(self):
        for _ in range(2):
            self.cnt = 0
            self.scan()
            self.random_search()
            self.symetry()
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
