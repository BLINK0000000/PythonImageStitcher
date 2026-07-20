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
    
    match1, match2 = rand_matches

    point1 = match1[0]
    point2 = match1[1]
    point3 = match2[0]
    point4 = match2[1]

    points = [point1, point2, point3, point4]
    checkPoints = []

    leaveOut = len(points) - 1
    # first check p1 p2 p3
    # 2nd check p1 p2 p4
    # 3rd check p1 p3 p4
    # 4th check p2 p3 p4
    # use 2 pointer method to identify which to leave out
    # use while loop to and have early return when points_are_collinear is true

    while leaveOut >= 0:
        for i in range(len(points)):
            if i != leaveOut:
                checkPoints.append(points[i])
            
        if points_are_collinear(checkPoints[0], checkPoints[1], checkPoints[2]):
            return True
        
        leaveOut -= 1
        
    return False


def points_are_collinear(p1, p2, p3):
    triagArea = 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

    return triagArea < 1e-5

def random_select_four_matches(matched_corner_pairs):
    maxIndex = len(matched_corner_pairs) - 1
    distinct = False
    match1 = None
    match2 = None

    while not distinct:
        matching = False

        match1 = matched_corner_pairs[np.random.randint(maxIndex)]
        match2 = matched_corner_pairs[np.random.randint(maxIndex)]

        point1 = match1[0]
        point2 = match1[1]
        point3 = match2[0]
        point4 = match2[1]

        points = [point1, point2, point3, point4]

        for i in range(len(points)):
            if matching:
                break

            for j in range(len(points)):
                if i != j:
                    if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                        matching = True
                        break

        if not matching:
            distinct = True
        

    return (match1, match2)


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

    matchedCorners = np.load("filtered_matched_corner_pairs.npy")

    # testH = np.load("transformation_matrix.npy")

    left_image = np.array(Image.open("left_image.png").convert('L'))
    right_image = np.array(Image.open("right_image.png").convert('L'))
    # final_warped_image = np.array(Image.open("final_warped_image.png").convert('L'))

    # best_transformation_matrix = np.array([[-0.014,    0.0,    1.0],
    #                                    [   0.0, -0.014,    0.0],
    #                                    [   0.0,    0.0, -0.014]])
    
    warped_image = inverse_warping(ransac_algorithm(matchedCorners), left_image, right_image)

    img = Image.fromarray(warped_image)
    img.show()

if __name__ == "__main__":
    main()

