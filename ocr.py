import numpy as np
import os
from PIL import Image             #to create image
import keras_ocr                  #text detection library
import matplotlib.pyplot as plt 
import cv2



def recognize_text(temp_path, input_path):
    file_name = os.path.basename(input_path)
    output_path = os.path.join('data', 'output', file_name)  #to get the file name from input path  
    image = keras_ocr.tools.read(temp_path) #image_path contains path of preprocessed image
    pipeline = keras_ocr.pipeline.Pipeline()
    predictions = pipeline.recognize([image])

    #code to create output image by masking everything except detected text.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    axs = [ax]

    for ax, image, prediction in zip(axs, [image], predictions):
        masked_image = np.ones_like(image) * 255
        for _, box in prediction:
            mask = np.zeros_like(image, dtype=np.uint8)
            polygon = np.array([box], dtype=np.int32)
            cv2.fillPoly(mask, polygon, (255, 255, 255))
            masked_image = np.where(mask == 255, image, masked_image)

        imagez = Image.fromarray(masked_image)
        imagez.save(output_path)

        #ax.imshow(masked_image)
        ax.axis('off')

    os.remove(temp_path)
    #plt.show()
