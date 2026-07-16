import numpy as np
from PIL import Image

def compute_NCC_component(image, corner_list, window_size):
    cornerComponents = []

    for corner in corner_list:
        
        # Finding the boundaries which the window covers
        yPosBoundary = corner[0] + np.floor(window_size / 2)
        yNegBoundary = corner[0] - np.floor(window_size / 2)
        xPosBoundary = corner[1] + np.floor(window_size / 2)
        xNegBoundary = corner[1] - np.floor(window_size / 2)

        sum = np.int64(0)
        patch = np.zeros((window_size, window_size)) 

        # Only start to compute NCC if the window lies within the boundaries of the image
        if (yPosBoundary < len(image) and yNegBoundary >= 0 and xPosBoundary < len(image[0]) and xNegBoundary >= 0):

            k = 0 # secondary manual iterator for the new patch (window of the image)
            # iterating between the boundaries of the window
            for i in range(int(yNegBoundary), int(yPosBoundary) + 1): 
                m = 0
                for j in range(int(xNegBoundary), int(xPosBoundary) + 1):
                    patch[k, m] = image[i, j]
                    sum += image[i, j]

                    m += 1

                k += 1

            mean = sum / (window_size**2)

            zeroMean = patch - mean # numerator of NCC

            SSD = np.sqrt(np.sum(zeroMean**2)) # denominator of NCC

            # Only compute NCC component if the denominator is not 0
            if (SSD != 0):
                component = zeroMean / SSD
                cornerComponents.append((corner[0], corner[1], component))
    
    return cornerComponents


def perform_feature_matching(NCCs_left, NCCs_right):
    best_matches = []
    second_best_matches = []

    # Checking all corner of the right image against one corner from the left, then moving on to the next left corner
    # and doing it all again (brute force)
    for componentL in NCCs_left:
        # resetting bestMatches
        best_match = (None, None, -1) # last element (NCC component) set to the maximum worst score to make sure
                                      # the very first best match is actually stored
        second_best_match = (None, None, -1)

        for componentR in NCCs_right:

            # Computing NCC score
            nccScore = np.sum(componentL[2] * componentR[2])

            if (nccScore > best_match[2]): # if score is better than current best score
                second_best_match = best_match 
                best_match = ((componentL[0], componentL[1]), (componentR[0], componentR[1]), nccScore)
            
            elif (nccScore > second_best_match[2]): # if score is better than the current second best score
                second_best_match = ((componentL[0], componentL[1]), (componentR[0], componentR[1]), nccScore)

        best_matches.append(best_match)
        second_best_matches.append(second_best_match)

    return best_matches, second_best_matches


def filter_corner_pairs_using_NCC_ratio(best_matching_corner_list, second_best_matching_corner_list):
    filteredMatchedCorners = []

    # Best matching and second matching will always have the same length and should correspond to eachother
    for i in range(len(best_matching_corner_list)):
        nccRatio = second_best_matching_corner_list[i][2] / best_matching_corner_list[i][2]

        if (nccRatio > 0 and nccRatio < 0.9):
            filteredMatchedCorners.append((best_matching_corner_list[i][0], best_matching_corner_list[i][1]))

    
    return filteredMatchedCorners
        
def main():
    left_image = np.array(Image.open("left_image.png").convert('L'))
    right_image = np.array(Image.open("right_image.png").convert('L'))

    left_corner_list = np.load("step4_left_nms_corner_response.npy")
    right_corner_list = np.load("step4_right_nms_corner_response.npy")
    
    window_size = 15

    # Assuming both images are same size for all cases (in this case they are)
    #imageHeight = 200
    #imageWidth = 210

    NCCs_left_component = compute_NCC_component(left_image, left_corner_list, window_size)
    NCCs_right_component = compute_NCC_component(right_image, right_corner_list, window_size)

    best_matching_corner_list = perform_feature_matching(NCCs_left_component, NCCs_right_component)[0]
    second_best_matching_corner_list = perform_feature_matching(NCCs_left_component, NCCs_left_component)[1]

    filtered_corner_pairs = filter_corner_pairs_using_NCC_ratio(best_matching_corner_list, second_best_matching_corner_list)


    # For testing
    test_NCCs_left_component = np.load("ncc_left_component.npy", allow_pickle=True)
    test_NCCs_right_component = np.load("ncc_right_component.npy", allow_pickle=True)

    test_best_matching_corner_list = np.load("best_matching_corner_list.npy", allow_pickle=True)
    test_second_best_matching_corner_list = np.load("second_best_matching_corner_list.npy", allow_pickle=True)

    
    print(filtered_corner_pairs)

    
    

if __name__ == "__main__":
    main()

