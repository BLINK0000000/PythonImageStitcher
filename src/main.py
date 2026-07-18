import featureMatching as ncc 
import harrisCornerPipeline as harris
import warping as warp

from PIL import Image
import numpy as np

def main():

    left_image = np.array(Image.open("left_image.png").convert('L'))
    right_image = np.array(Image.open("right_image.png").convert('L'))

    left_imageSize = np.shape(left_image)
    image_height = left_imageSize[0]
    image_width = left_imageSize[1]

    left_image = harris.gaussian_filtering(left_image, image_width, image_height)
    right_image = harris.gaussian_filtering(right_image, image_width, image_height)

    left_ix_square, left_iy_square, left_ixiy = harris.compute_image_derivatives(left_image, image_width, image_height)
    right_ix_square, right_iy_square, right_ixiy = harris.compute_image_derivatives(right_image, image_width, image_height)
    alpha = 0.04
    threshold = 10 ** 6

    cornerness_score_left = harris.compute_cornerness_score(left_ix_square, left_iy_square, left_ixiy, alpha, threshold, image_width, image_height)
    cornerness_score_right = harris.compute_cornerness_score(right_ix_square, right_iy_square, right_ixiy, alpha, threshold, image_width, image_height)

    left_corner_response = harris.build_corner_response(cornerness_score_left, image_width, image_height)
    right_corner_response = harris.build_corner_response(cornerness_score_right, image_width, image_height)

    nms_left_response = harris.non_maximum_suppression(left_corner_response, image_width, image_height)
    nms_right_response = harris.non_maximum_suppression(right_corner_response, image_width, image_height)

    window_size = 15

    left_corner_list = nms_left_response
    right_corner_list = nms_right_response

    NCCs_left_component = ncc.compute_NCC_component(left_image, left_corner_list, window_size)
    NCCs_right_component = ncc.compute_NCC_component(right_image, right_corner_list, window_size)

    best_matching_corner_list = ncc.perform_feature_matching(NCCs_left_component, NCCs_right_component)[0]
    second_best_matching_corner_list = ncc.perform_feature_matching(NCCs_left_component, NCCs_left_component)[1]

    filtered_corner_pairs = ncc.filter_corner_pairs_using_NCC_ratio(best_matching_corner_list, second_best_matching_corner_list)

    np.random.seed(784)

    best_transformation_matrix = warp.ransac_algorithm(filtered_corner_pairs)
    
    warped_image = warp.inverse_warping(best_transformation_matrix, left_image, right_image)

    img = Image.fromarray(warped_image)
    img.show()

if __name__ == "__main__":
    main()

