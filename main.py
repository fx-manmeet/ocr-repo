from preprocess import preprocess_image_for_ocr
from ocr import recognize_text

def main(image_path):
    preprocessed_image = preprocess_image_for_ocr(image_path)
    recognize_text('final_image.png')

if __name__ == '__main__':
    image_path = '.\\imgs\\1708521114790496.jpg'
    main(image_path)
