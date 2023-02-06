import numpy as np
import cv2 as cv


def gradn(im):
    """
    Compute the norm of the gradient of the image im of shape (m, n, 3). Returns an array of shape (m - 1, n - 1).
    """
    grad = np.sqrt(np.diff(im, axis=0)[:, :-1]**2 + np.diff(im, axis=1)[:-1, :]**2)
    return grad


def compute_mask_1(vect_field, m, n, p):
    """
    Compute the mask of the copy-moved area from the vect_field. Method 1.
    """
    r = p
    th = 0.5
    th_comp = 0.05
    s = 2 * p

    #Compute the gradn of x and y displacement map
    vx = gradn(vect_field[..., 0])
    vy = gradn(vect_field[..., 1])

    #Compute a first mask
    mask_0 = np.zeros((m, n))
    u = (np.mean(vx) + np.mean(vy)) / 100
    mask_0[:-1, :-1] = 1 * (vy < u) * (vx < u)

    #Filter a big part of the noise
    kernel = np.ones((r, r))
    kernel = kernel / np.sum(kernel)
    mask_1 = cv.filter2D(mask_0, -1, kernel)
    mask_2 = 1 * (mask_1 > th)

    #look at connexe component to keep just biggest ones
    mask_3 = np.uint8((mask_2))
    N, component = cv.connectedComponents(mask_3)
    white_pixel = np.sum(mask_2)
    
    liste_component = []
    for i in range(1,N):
        #we keep just component which as more than 5% of white pixels
        if np.sum(1 * (component == i)) / white_pixel > th_comp:
            liste_component.append(i)
    mask_4 = np.zeros((m, n))
    for i in liste_component:
        mask_4 += 1 * (component == i)

    #dilatate the result to compensate the patch effect
    kernel = np.ones((s, s))
    mask = cv.dilate(mask_4, kernel) > 0
    return mask


def compute_mask_2(vect_field, m, n, p):
    """
    Compute the mask of the copy-moved area from the vect_field. Method 2.
    """
    # Compute end_points = start_points + displacement_vectors
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    ij = np.stack((ii, jj), axis=-1)
    end_points = ij + vect_field
    # Compose function f : start_points -> end_points with itself
    end_points2 = end_points[end_points[..., 0], end_points[..., 1]]
    # Compute the ground distances after this second application of f
    back_and_forth_distance = np.max(np.abs(end_points2 - ij), axis=-1)
    # mask := "coherent" points = points where back_and_forth_distance is 0
    mask = (back_and_forth_distance == 0).astype("uint8")
    # erode mask
    eroded_mask = cv.erode(mask, np.ones((2, 2))) 
    # Select the 2 biggest connected components
    N, components = cv.connectedComponents(eroded_mask)
    bc = np.bincount(components.flatten())
    indices = np.argsort(bc)[::-1]
    biggest_components = ((components == indices[1]) + (components == indices[2])).astype("uint8")
    # Dilate mask
    final_mask = cv.dilate(biggest_components, np.ones((15, 15))) > 0
    return final_mask


def fscore(mask, gt):
    """Compute F-measure of computed mask vs ground truth."""
    tp = np.sum(mask * gt)
    fp = np.sum(mask * (gt < 1))
    fn = np.sum((mask < 1) * gt)
    
    return 2 * np.sum(tp) / (2 * np.sum(tp) + np.sum(fn) + np.sum(fp))
