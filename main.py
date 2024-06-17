# from preprocess import preprocess_image_for_ocr
# from ocr import recognize_text

# def main(input_path):
#     preprocess_image_for_ocr(input_path) #preprocess the image
#     recognize_text('final_image.png',input_path) #detect text from final_image.png(it is preprocessed image)

# if __name__ == '__main__':
#     image_path = 'data\\input\\1708520917612554.jpg'    #input image path
#     main(image_path)   

import argparse    
import os          #for file paths 
from preprocess import preprocess_image_for_ocr
from ocr import recognize_text

def main(input_path):
    temp_path = preprocess_image_for_ocr(input_path) # preprocess the image
    recognize_text(temp_path,input_path) # detect text from final_image.png (preprocessed image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image for OCR.')
    parser.add_argument('image_path', type=str, help='Path to the input image')

    args = parser.parse_args()
    image_path = os.path.normpath(args.image_path)  # Normalize the path for the current OS
    main(image_path)


# argparse ..
# temp-img_name
# temp folder delete after result.
# utilise os.path.join
# file splitting                               
