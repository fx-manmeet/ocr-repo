import numpy as np
from PIL import Image
import keras_ocr
import matplotlib.pyplot as plt
import cv2



def recognize_text(image_path):
    image = keras_ocr.tools.read(image_path)
    pipeline = keras_ocr.pipeline.Pipeline()
    predictions = pipeline.recognize([image])

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
        imagez.save('ocr-ed_image.png')

        #ax.imshow(masked_image)
        ax.axis('off')

    #plt.show()
