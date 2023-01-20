import numpy as np
import cv2 as cv

def gradn(im):
    """
    Compute the norm of the gradient of the image im of shape (m,n,3). The return is of size (m-1,n-1)
    """
    grad = np.sqrt( (np.diff(im,axis=0)[:,:-1])**2 + (np.diff(im,axis=1)[:-1,:])**2 )
    return grad

def compute_mask(vect_field, n, m, p):
    """
    Compute the mask of copy-move with the vect_field. It return the computed mask and the number of detection.
    """
    r = p
    th = 0.5
    th_comp = 0.05
    s = 2*p
    #Compute the gradn of x and y displacement map
    vx = gradn(vect_field[..., 0])
    vy = gradn(vect_field[..., 1])

    #Compute a first mask
    mask_0 = np.zeros((m,n))
    u = (np.mean(vx)+np.mean(vy))/100
    mask_0[:-1,:-1] = 1*(vy<u)*(vx<u)

    #Filter a big part of the noise
    kernel = np.ones((r,r))
    kernel = kernel /np.sum(kernel)
    mask_1 = cv.filter2D(mask_0, -1, kernel) 
    mask_2 = 1*(mask_1>th)

    #look at connexe component to keep just biggest ones
    mask_3 = np.uint8((mask_2))
    N, component = cv.connectedComponents(mask_3)
    white_pixel = np.sum(mask_2)
    
    Liste_component = []
    for i in range(1,N):
        #we keep just component which as more than 5% of white pixels
        if np.sum(1*(component==i))/white_pixel > th_comp:
            Liste_component.append(i)
    mask_4 = np.zeros((m,n))
    number_detection = len(Liste_component)//2
    for i in Liste_component:
        mask_4 += 1*(component==i)

    #dilatate the result to compensate the patch effect
    kernel = np.ones((s,s))
    mask = cv.dilate(mask_4, kernel)
    return mask, number_detection