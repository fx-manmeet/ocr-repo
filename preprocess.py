import cv2
import os

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

def preprocess_image_for_ocr(input_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = apply_blur(image)
    thresholded_image = apply_threshold(blurred_image)
    morphed_image = apply_morphological_operations(thresholded_image)
    final_image = remove_noise(morphed_image)

    file_name = os.path.basename(input_path)
    output_path = os.path.join('data', 'temp', f'temp-{file_name}')
    print(output_path)
    save_image(final_image, output_path)       #saves preprocessed image to be fed into detection pipeline
    return output_path
