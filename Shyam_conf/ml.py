from tensorflow import keras
import tensorflow as tf
import os
import cv2
import numpy as np
model = keras.models.load_model(r"C:\Users\sbgor\OneDrive\Desktop\Confident_level_prediction_final.h5", compile=False)
model.compile()

img = cv2.imread(r"C:\Users\sbgor\OneDrive\Desktop\nr.jpg")

resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat >= 0.55:
    print(f'Predicted class is confident')
elif(yhat<=0.20):
    print(f'Predicted class is unconfident')
else:
    print(f'prediction class is neutral')