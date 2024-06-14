from preprocess import preprocess_image_for_ocr
from ocr import recognize_text

def main(input_path):
    preprocessed_image = preprocess_image_for_ocr(input_path)
    recognize_text('final_image.png',input_path)

if __name__ == '__main__':
    image_path = '.\\imgs\\1708521114790496.jpg'
    main(image_path)
