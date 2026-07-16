import numpy as np
from PIL import Image

def round_matrix(arr: np.ndarray) -> np.ndarray: 
    return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(np.int64)

def thresholdingCorners(C):
    threshold = 1500000

    return (C * (C > threshold))

def getCornerResponse(IxSq, IxIy, IySq):
    alpha = 0.04

    H = np.zeros((2, 2))

    C = np.zeros(np.shape(IxSq))

    IxSq = np.pad(IxSq, 1, 'edge')
    IySq = np.pad(IySq, 1, 'edge')
    IxIy = np.pad(IxIy, 1, 'edge')

     # 2 increment variables, one for image and one for kernel (manual counting or zip)
    fi = 0
    for i in range(1, len(IxIy) - 1):
        fj = 0
        for j in range(1, len(IxIy[0]) - 1):
            sumIxIy = 0
            sumIxSq = 0
            sumIySq = 0

            for k in range(i - 1, i + 2):
                for m in range(j - 1, j + 2):

                    sumIxIy += IxIy[k, m]
                    sumIxSq += IxSq[k, m]
                    sumIySq += IySq[k, m]
            
            H[0, 0] = sumIxSq
            H[0, 1] = sumIxIy
            H[1, 0] = sumIxIy
            H[1, 1] = sumIySq

            H = round_matrix(H)

            C[fi, fj] = (np.linalg.det(H)) - (alpha * ((np.trace(H))**2))
            fj += 1

        fi += 1


    return round_matrix(C)
    

def gaussian_filtering(image, image_width, image_height):
    paddedImage = np.pad(image, 1, 'edge') # border boundary padding image

    # 3x3 Gaussian filter kernel 
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    kernel = kernel / 16

    # initializing results array
    result = np.zeros((image_height, image_width))

    # 4 increment variables, image, sliding window, kernel and result matrix (manual counting for last 3)
    # Will iterate within the bounds of the original image, not over the padded pixel values, but using a loop below
    # to implement a sliding window will include the padded values without throwing errors

    fi = 0 # row index for resulting image matrix
    for i in range(1, len(paddedImage) - 1): # row index for padded image

        fj = 0 # column index for resulting image matrix

        for j in range(1, len(paddedImage[0]) - 1): # column index for padded image
            sum = 0 # Holding the sum of values from applying the kernel matrix

            ki = 0 # row index for the kernel

            # Iterating within a bound of 3x3 to implement the sliding window
            # i.e. i is the pixel value we want to evaluate (centre pixel) and we will compute using
            #      the surrounding pixels in a 3x3 "box" by iterating from i-1 to i+2 (range of 3)
            #      and so on with the j index.

            for k in range(i - 1, i + 2): # loop through the rows within a range of 3

                kj = 0 # column index for the kernel, this will reset after we finish computing with the 
                       # current pixel.
                for m in range(j - 1, j + 2): # loop through the columns within a range of 3

                    sum += paddedImage[k, m] * kernel[ki, kj]

                    kj += 1 # move on to next column in the window
                    
                ki += 1 # move on to the next row in the window
            
            result[fi, fj] = sum

            fj += 1

        fi += 1

    result = np.array(result).astype(np.float128)

    return result

def compute_image_derivatives(image, image_width, image_height):
    paddedImage = np.pad(image, 1, 'edge') # border boundary padding image

    vertSobelKernel = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
    vertSobelKernel = vertSobelKernel / 8

    horzSobelKernel = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
    horzSobelKernel = horzSobelKernel / 8

    Ix = np.zeros((image_height, image_width))
    Iy = np.zeros((image_height, image_width))

    # 4 increment variables, image, sliding window, kernel and derivative matrices (manual counting for last 3)
    # Will iterate within the bounds of the original image, not over the padded pixel values, but using a loop below
    # to implement a sliding window will include the padded values without throwing errors

    fi = 0 # row index for derivative matrices
    for i in range(1, len(paddedImage) - 1): # row index for padded image
        fj = 0 #
        for j in range(1, len(paddedImage[0]) - 1):
            sumVertSobel = 0 # holding sum from applying the vertical sobel kernel
            sumHorzSobel = 0 # same as above but for horizontal kernel

            ki = 0 # row index for kernels

            for k in range(i - 1, i + 2):
                kj = 0 # column index for kernels
                for m in range(j - 1, j + 2):

                    sumVertSobel += paddedImage[k, m] * vertSobelKernel[ki, kj]
                    sumHorzSobel += paddedImage[k, m] * horzSobelKernel[ki, kj]

                    kj += 1
                    
                ki += 1
            
            Ix[fi, fj] = sumVertSobel
            Iy[fi, fj] = sumHorzSobel

            fj += 1

        fi += 1

    IxSq = Ix**2 # computing Ix_squared
    IySq = Iy**2 # Iy squared
    IxIy = Ix * Iy

    return IxSq, IySq, IxIy


def compute_cornerness_score(ix_square, iy_square, ixiy, alpha, threshold, image_width, image_height):
    H = np.zeros((2, 2))

    cornerList = []

    # Boundary padding derivative matrices
    IxSq = np.pad(ix_square, 1, 'edge')
    IySq = np.pad(iy_square, 1, 'edge')
    IxIy = np.pad(ixiy, 1, 'edge')

    # 3 increment variables, one for derivatives, 3x3 window, and coordinates of the centre pixel 
    # values we evaluate (manual counting for last 2)

    fi = 0 # row index for centre pixel value

    # Looping over non padded pixels in deriative matrices
    for i in range(1, len(IxIy) - 1):
        fj = 0
        for j in range(1, len(IxIy[0]) - 1):
            sumIxIy = 0
            sumIxSq = 0
            sumIySq = 0

            # Looping over derivative matrices within a 3x3 window
            for k in range(i - 1, i + 2):
                for m in range(j - 1, j + 2):
                    
                    # Summing derivative values within the window
                    sumIxIy += IxIy[k, m]
                    sumIxSq += IxSq[k, m]
                    sumIySq += IySq[k, m]
            
            # Creating H matrix
            # H = [IxSq IxIy
            #      IxIy IySq]
            H[0, 0] = sumIxSq
            H[0, 1] = sumIxIy
            H[1, 0] = sumIxIy
            H[1, 1] = sumIySq

            cornerScore = (np.linalg.det(H)) - (alpha * ((np.trace(H))**2))

            # Setting a threshold to only select high cornerness scores,
            # this is to ensure that these are definitely corners
            if (cornerScore > threshold):
                # Add to corner list if score is over threshold and store the coordinates of that pixel too
                cornerList.append((fi, fj, cornerScore))

            fj += 1

        fi += 1

    cornerList.sort(key=lambda score: score[:][2], reverse=True) # Sorting the list of corners and coordinates based on the corner score in descending order

    return cornerList[0:1000] # Only need the top 1000 scores


def non_maximum_suppression(corner_response, image_width, image_height):
    C = np.pad(corner_response, 1, 'edge') # border boundary padding matrix of corner scores
    suppressedCornerList = [] # list of coordinates of corner scores after non-maximum suppression

    # 3 increment variables, one for corner score matrix, sliding window, and coordinates of centre pixel
    # values we evaluate (manual counting for last 2)

    fi = 0 # row index of centre pixel value

    # Looping over the non padded corner scores
    for i in range(1, len(C) - 1):
        fj = 0
        for j in range(1, len(C[0]) - 1):

            windowVals = np.array([]) # array to store all corner values within 3x3 window

            if (C[i, j] != 0): # only perform suppression if the centre corner/pixel value is actually a corner
                for k in range(i - 1, i + 2):
                    for m in range(j - 1, j + 2):

                        windowVals = np.append(windowVals, C[k, m])

                # If the centre corner/pixel value is the maximum/highest value in the 3x3 window, keep it as a corner
                if (np.max(windowVals) == C[i, j]):
                    suppressedCornerList.append((fi, fj)) # Add the corner's coordinates to the corner list

            fj += 1

        fi += 1

    return suppressedCornerList

def main():

    left_image = np.array(Image.open("left_image.png").convert('L'))
    right_image = np.array(Image.open("right_image.png").convert('L'))

    left_imageSize = np.shape(left_image)
    image_height = left_imageSize[0]
    image_width = left_imageSize[1]

    left_image = gaussian_filtering(left_image, image_width, image_height)
    right_image = gaussian_filtering(right_image, image_width, image_height)

    left_ix_square, left_iy_square, left_ixiy = compute_image_derivatives(left_image, image_width, image_height)
    right_ix_square, right_iy_square, right_ixiy = compute_image_derivatives(right_image, image_width, image_height)
    alpha = 0.04
    threshold = 10 ** 6

    cornerness_score_left = compute_cornerness_score(left_ix_square, left_iy_square, left_ixiy, alpha, threshold, image_width, image_height)
    cornerness_score_right = compute_cornerness_score(right_ix_square, right_iy_square, right_ixiy, alpha, threshold, image_width, image_height)

    left_corner_response = np.load("step4_left_corner_response.npy")
    right_corner_response = np.load("step4_right_corner_response.npy")

    nms_left_response = non_maximum_suppression(left_corner_response, image_width, image_height)
    nms_right_response = non_maximum_suppression(right_corner_response, image_width, image_height)


if __name__ == "__main__":
    main()

