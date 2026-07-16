import numpy as np
from PIL import Image


def compute_homo(matched_corner_pairs):
    """
    Compute the homography matrix on matched corners
    
    arguments:
        matched_corner_pairs    : list of tuples containing coordinates of 
                                  matching corners from the left image and 
                                  right image.
    
    returns:
        H                       : 2D 3x3 numpy array containing the homography
                                  transformation matrix.
    """
    
    A = np.zeros((len(matched_corner_pairs) * 2, 9))

    i = 0
    for pair in matched_corner_pairs:
        
        A[i, 3] = pair[0, 1]
        A[i, 4] = pair[0, 0]
        A[i, 5] = 1

        A[i + 1, 0] = pair[0, 1]
        A[i + 1, 1] = pair[0, 0]
        A[i + 1, 2] = 1

        A[i, 8] = -(pair[1, 0])
        A[i + 1, 8] = -(pair[1, 1])

        A[i, 6] = A[i, 8] * A[i, 3]
        A[i, 7] = A[i, 8] * A[i, 4]

        A[i + 1, 6] = A[i + 1, 8] * A[i, 3]
        A[i + 1, 7] = A[i + 1, 8] * A[i, 4]

        i += 2

    U, D, Vt = np.linalg.svd(A)
    
    # Extracting last row of Vt since it contains the smallest singular values
    # which minimises A, and reshaping into 3x3 to get homography.
    H = Vt[-1].reshape((3,3))

    return H

def checking_points_are_collinear(rand_matches):
    pass

def random_select_four_matches(matched_corner_pairs):
    pass

def ransac_algorithm(matched_corner_pairs):
    """
    Using RANSAC to find the best homography matrix of given matched corners
    
    arguments:
        matched_corner_pairs    : list of tuples with coordinates of matching 
                                  corners in the left and right image
                                          
    returns:
        bestH                   : best homography transformation which had the
                                  most inliers
    """
    lastTopInliers = 0
    inlierPairs = []
    bestInlierPairs = []
    iterations = 1500
    allowableError = 2 # pixels
    
    for i in range(iterations):

        batchCorners = random_select_four_matches(matched_corner_pairs)

        if (checking_points_are_collinear(batchCorners) == False):
            H = compute_homo(batchCorners)
            numOfInliers = 0
            inlierPairs.clear()
            
            for pair in matched_corner_pairs:
                rightPoint = np.array([pair[1, 1], pair[1, 0], 1])
                leftPoint = np.array([pair[0, 1], pair[0, 0], 1])
                transformedLeftPoint = H @ leftPoint
                transformedLeftPoint /= transformedLeftPoint[2]
                
                d = np.sqrt((rightPoint[0] - transformedLeftPoint[0])**2 + (rightPoint[1] - transformedLeftPoint[1])**2)
                
                if (d <= allowableError):
                    numOfInliers += 1
                    inlierPairs.append(pair)
                
            if (numOfInliers > lastTopInliers):
                lastTopInliers = numOfInliers
                bestInlierPairs = inlierPairs
    
    bestH = compute_homo(bestInlierPairs)
    
    return bestH 


def inverse_warping(best_transformation_matrix, left_image, right_image):
    """
    Using simplified inverse warping to stitch two related images together
    
    arguments:
        best_transformation_matrix  : Best homography transformation found from
                                      RANSAC using matching corners of images
        
        left_image                  : 2D numpy array of the left grayscale image
        right_image                 : 2D numpy array of the right grayscale
                                      image
                                      
    returns:
        canvas                      : Final stitched/warped image
    """
    blendWeight = 0.95 # for linear colour blending
    
    leftImageWidth = len(left_image[0])
    rightImageWidth = len(right_image[0])
    canvasWidth = leftImageWidth * 2 # Assuming both images are same width
    
    leftImageHeight = len(left_image)
    rightImageHeight = len(right_image)
    canvasHeight = leftImageHeight

    canvas = np.zeros((canvasHeight, canvasWidth))

    for i in range(len(canvas)):
        for j in range(len(canvas[0])):
            canvasCoord = np.array([j, i, 1])

            mappedCoord = best_transformation_matrix @ canvasCoord
            mappedCoord /= mappedCoord[2]
            
            # source coords in left image and mapped coords outside of right image
            if (j < leftImageWidth and i < leftImageHeight and mappedCoord[0] < 0):
                canvas[i, j] = left_image[i, j]
                
                
            # source coords in left image and mapped coords inside right image
            elif (j < leftImageWidth and i < leftImageHeight and mappedCoord[0] >= 0 and mappedCoord[0] < rightImageWidth - 1 and mappedCoord[1] < rightImageHeight - 1 and mappedCoord[1] > 0):
                colour1 = left_image[i, j]
                
                # billinear interpolation
                f12 = [int(np.floor(mappedCoord[0])), int(np.floor(mappedCoord[1]))]
                f22 = [int(np.ceil(mappedCoord[0])), int(np.floor(mappedCoord[1]))]
                f11 = [int(np.floor(mappedCoord[0])), int(np.ceil(mappedCoord[1]))]
                f21 = [int(np.ceil(mappedCoord[0])), int(np.ceil(mappedCoord[1]))]

                a = mappedCoord[0] - f12[0]
                b = mappedCoord[1] - f12[1]
                c = 1.0 - a
                d = 1.0 - b

                colour2 = (c * d * right_image[f12[1], f12[0]] 
                           + a * d * right_image[f22[1], f22[0]] 
                           + b * c * right_image[f11[1], f11[0]] 
                           + a * b * right_image[f21[1], f21[0]])

                colourBlend = colour1 * blendWeight + colour2 * (1.0 - blendWeight)

                canvas[i, j] = colourBlend

            # source coords outside left image and mapped coords inside right image
            elif (j >= leftImageWidth and mappedCoord[0] >= 0 and mappedCoord[0] < rightImageWidth - 1 and mappedCoord[1] > 0 and mappedCoord[1] < rightImageHeight - 1):

                # billinear interpolation
                f12 = [int(np.floor(mappedCoord[0])), int(np.floor(mappedCoord[1]))]
                f22 = [int(np.ceil(mappedCoord[0])), int(np.floor(mappedCoord[1]))]
                f11 = [int(np.floor(mappedCoord[0])), int(np.ceil(mappedCoord[1]))]
                f21 = [int(np.ceil(mappedCoord[0])), int(np.ceil(mappedCoord[1]))]

                a = mappedCoord[0] - f12[0]
                b = mappedCoord[1] - f12[1]
                c = 1.0 - a
                d = 1.0 - b

                colour2 = (c * d * right_image[f12[1], f12[0]] 
                           + a * d * right_image[f22[1], f22[0]] 
                           + b * c * right_image[f11[1], f11[0]] 
                           + a * b * right_image[f21[1], f21[0]])
                
                canvas[i, j] = colour2

    return canvas


def main():

    # Testing
    np.random.seed(784)

    matchedCorners = np.load("filtered_matched_corner_pairs.npy")

    testH = np.load("transformation_matrix.npy")

    left_image = np.array(Image.open("left_image.png").convert('L'))
    right_image = np.array(Image.open("right_image.png").convert('L'))
    final_warped_image = np.array(Image.open("final_warped_image.png").convert('L'))

    best_transformation_matrix = np.array([[-0.014,    0.0,    1.0],
                                       [   0.0, -0.014,    0.0],
                                       [   0.0,    0.0, -0.014]])
    
    warped_image = inverse_warping(best_transformation_matrix, left_image, right_image)

    img = Image.fromarray(warped_image)
    img.show()

if __name__ == "__main__":
    main()

