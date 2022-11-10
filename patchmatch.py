"""
m : image height
n : image length
p : 2*p+1 is the patch length
im : numpy array of float of size (m,n,3) : the image
vect_field : numpy array of size (m,n,2) : field of displacement vector for each pixel
T : int : inferior limit for the infinite norm of displacements vectors
"""

import numpy as np
np.random.seed(0)

def euclidean_distance_patchs(im, i1, j1, i2, j2, p):
    """
    Compute the euclidean distance between patchs of size 2*p+1 center in (i1,j1) and (i2,j2)
    """
    patch_1 = im[i1-p:i1+p+1,j1-p:j1+p+1,:]
    patch_2 = im[i2-p:i2+p+1,j2-p:j2+p+1,:]
    return np.sqrt(np.sum((patch_1-patch_2)**2))

def create_vect_field(im, T, p):
    """
    A function to create a random vect_field where every vector are biger than T in infinite norm  
    Parameters
    ----------
    im : array-like, shape (m, n, 3)
        image
    T : int
        inferior limit for the infinite norm of displacements vectors
    
    Returns
    -------
    vect_field : array-like, shape (m, n, 3)
        Field of displacement vectors for each pixel. vect_field[i,j,0] is the x coordonate of the displacement vector. 
        vect_field[i,j,1] is the y coordonate of the displacement vector. 
        vect_field[i,j,2] is the distance between the patch and the moved patch.

    """
    m,n,_ = im.shape
    vect_field = np.zeros((m,n,3))

    #generate the target point with the constraint T
    I = np.arange(p, m-p)
    for i in range(m):
        idx = np.where(np.abs(I-i)>=T)
        vect_field[i,:,0] = np.random.choice(I[idx], n)

    J = np.arange(p, n-p)
    for j in range(n):
        idx = np.where(np.abs(J-j)>=T)
        vect_field[:,j,1] = np.random.choice(J[idx], m)

    #compute the displacement vectors
    pos = np.transpose(np.mgrid[0:m, 0:n], axes=(1, 2, 0))

    vect_field[:,:,0:2] = vect_field[:,:,0:2] - pos

    for i in range(m):
        for j in range(n):
            vect_field[i,j,2] = euclidean_distance_patchs(im, i, j, vect_field[i,j,0], vect_field[i,j,1])

    return vect_field

class PatchMatch:
    def __init__(self, im, T, p, N, L) -> None:
        """Instantiate the PatchMatch algorithm 
        
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
        self.vect_field = create_vect_field(im, T, p)
    
    def patch(self, i, j):
        """Return patch centered at (i, j)."""
        p = self.p
        return self.im[i-p:i+p+1, j-p:j+p+1]
    
    def dist(self, i, j, k, l):
        """Return l2 distance between patch centered at (i, j) and patch centered at (k, l)."""
        # print(i, j, k, l)
        return np.linalg.norm(self.patch(i, j) - self.patch(k, l))
    
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
                d0 = self.vect_field[i, j, 2]
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
                idx = np.argmin([d0, d_up, d_left])
                # Propagate best displacement
                if idx == 1:
                    self.vect_field[i, j, 0:2] = self.vect_field[i-1, j, 0:2]
                    self.vect_field[i, j, 2] = d_up
                if idx == 2:
                    self.vect_field[i, j, 0:2] = self.vect_field[i, j-1, 0:2]
                    self.vect_field[i, j, 2] = d_left
    
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
                    x, y = self.vect_field[i,j]
                    x_ = np.random.randint(np.max(x-2**(k-1),p), np.min(x+2**(k-1)+1,m-p))
                    y_ = np.random.randint(np.max(y-2**(k-1),p), np.min(y+2**(k-1)+1,m-p))
                    if self.test_min_separation(x_,y_):
                        d_init = self.vect_field[i,j,2]
                        d_test = self.dist(i,j,i+x_,j+y_)
                        if d_test < d_init:
                            self.vect_field[i,j] = np.array([x_,y_])

    def run(self):
        """Run the PatchMatch algorithm and return the resulting vector field."""
        for _ in range(2*self.N):
            self.scan()
            self.flip()
        return self.vect_field