import cv2
import matplotlib.pyplot as plt

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

def preprocess_image_for_ocr(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = apply_blur(image)
    thresholded_image = apply_threshold(blurred_image)
    morphed_image = apply_morphological_operations(thresholded_image)
    final_image = remove_noise(morphed_image)

    save_image(final_image, 'final_image.png')          #saves preprocessed image to be fed into detection pipeline
