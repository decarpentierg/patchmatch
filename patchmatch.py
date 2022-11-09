"""
m : image height
n : image length
p : 2*p+1 is the patch length
im : numpy array of float of size (m,n,3) : the image
vect_field : numpy array of size (m,n,2) : field of displacement vector for each pixel
"""

import numpy as np

def create_vect_field(im, T):
    """
    A function to create a random vect_field where every vector are biger than T in infinite norm    
    """
    m,n,_ = im.shape
    vect_field = np.random.randint(low = (0,0), high = (m-1,n-1), size = (n,m,2))

    #generate the target point with the constraint T
    interval_1 = np.arange(m)
    for i in range(m):
        vect_field[i,:,0] = np.random.choice(interval_1[np.abs(interval_1-i)>=T], n)

    interval_2 = np.arange(n)
    for j in range(n):
        vect_field[:,j,1] = np.random.choice(interval_2[np.abs(interval_2-j)>=T], m)

    #compute the displacement vectors
    pos = np.zeros((m,n,2))
    pos[:,:,0] = interval_1[:,None]*np.ones(n)[None,:]
    pos[:,:,1] = interval_2[None,:]*np.ones(m)[:,None]

    vect_field = vect_field - pos

    return vect_field





