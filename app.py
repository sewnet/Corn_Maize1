import streamlit as st
import tensorflow as tf
import os
import cv2
import PIL
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model


st.title("Corn Maize Classification")
st.header("Please input an image to be classified:")
#st.text("Created by SU")
    
uploaded_file = st.file_uploader("Upload an Image", type="jpg")

# Load the model
model = keras.models.load_model("LeafDisease_Corn_Maize-DenseNet121.h5")
opt = Adam(learning_rate= 0.0001)
model.compile(optimizer=opt, loss= 'categorical_crossentropy', metrics=['accuracy'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Convert image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array
    #st.write("HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    
#     res = model.evaluate(data)
#     st.write ("Loss and accuracy are:" + str(res))
    
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    st.write ("Predictions are:", prediction)
    st.write ("Predictions percentage:", prediction_percentage)
    


