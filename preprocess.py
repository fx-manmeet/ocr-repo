import cv2
import os
import numpy as np

def save_image(image, filename):
    cv2.imwrite(filename, image)

def apply_blur(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def apply_threshold(image):
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    return thresholded

def apply_morphological_operations(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def remove_noise(image):
    noise_removed = cv2.medianBlur(image, 5)
    return noise_removed

def apply_sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9,-1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def preprocess_image_for_ocr(input_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = apply_blur(image)
    thresholded_image = apply_threshold(blurred_image)
    morphed_image = apply_morphological_operations(thresholded_image)
    denoised_image = remove_noise(morphed_image)
    final_image = unsharp_mask(denoised_image)

    file_name = os.path.basename(input_path)
    output_path = os.path.join('data', 'temp', f'temp-{file_name}')
    print(output_path)
    save_image(final_image, output_path)       #saves preprocessed image to be fed into detection pipeline
    return output_path
