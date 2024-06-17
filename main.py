from preprocess import preprocess_image_for_ocr
from ocr import recognize_text

def main(input_path):
    preprocess_image_for_ocr(input_path) #preprocess the image
    recognize_text('final_image.png',input_path) #detect text from final_image.png(it is preprocessed image)

if __name__ == '__main__':
    image_path = 'data\\input\\1708520932181342.jpg'    #input image path
    main(image_path)                                
