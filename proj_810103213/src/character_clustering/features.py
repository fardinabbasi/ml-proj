import numpy as np

from skimage.feature import hog
from .data import custom_resize
from scipy.ndimage import center_of_mass
from skimage.filters import threshold_otsu

def extract_hog_features(char_images, image_size = (32, 32)):

    features_list = []
    print(f"Extracting HOG features for {len(char_images)} images...")

    for img in char_images:
        # grayscale and resize
        gray_pil_img = img.convert('L')
        resized_img = custom_resize(gray_pil_img, image_size)
        resized_array = np.array(resized_img)

        hog_features_vec = hog(resized_array, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')

        features_list.append(hog_features_vec)

    print("HOG feature extraction complete.")
    return np.array(features_list)


def extract_raw_pixel_features(char_images, image_size = (32, 32)):

    features_list = []
    print(f"Extracting Raw Pixel features for {len(char_images)} images...")

    for img in char_images:
        # grayscale and resize
        gray_pil_img = img.convert('L')
        resized_img = custom_resize(gray_pil_img, image_size)

        # Convert to numpy array
        resized_array = np.array(resized_img)

        pixel_features_vec = resized_array.flatten()

        features_list.append(pixel_features_vec)

    print("Raw Pixel feature extraction complete.")
    return np.array(features_list)

def extract_geometric_features(char_images):

    features_list = []
    print(f"Extracting Geometric features for {len(char_images)} images...")

    for img in char_images:
        gray_img = img.convert('L')
        img_array = np.array(gray_img)


        threshold = threshold_otsu(img_array)
        binary_array = (img_array < threshold).astype(int)
        
        height, width = binary_array.shape


        aspect_ratio_raw = width / height
        sigmoid_aspect_ratio = 1 / (1 + np.exp(-aspect_ratio_raw))
        
        # if aspect_ratio_raw < 0.75:
        #     discretized_aspect_ratio = 0  # Vertical
        # elif aspect_ratio_raw > 1.33:
        #     discretized_aspect_ratio = 2  # Horizontal
        # else:
        #     discretized_aspect_ratio = 1  # Square-ish

        pixel_density = np.mean(binary_array)
        center_y, center_x = center_of_mass(binary_array)
        norm_center_y = center_y / height
        norm_center_x = center_x / width
        
        # Assemble the feature vector with the new discretized feature
        feature_vector = [sigmoid_aspect_ratio, pixel_density, norm_center_y, norm_center_x]
        features_list.append(feature_vector)

    print("Geometric feature extraction complete.")
    return np.array(features_list)
