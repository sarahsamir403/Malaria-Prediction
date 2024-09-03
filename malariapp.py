import streamlit as st
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Malaria Classification")
st.write("Malaria Prediction ")

#upload an image
uploaded_image = st.file_uploader("Upload an Image", type = ['jpg', 'png', 'jpeg'])

#load model
Model = load_model("D:\Project_malaria\Malria_model.h5")
st.write("*note*")
st.write("class 0 is Infected")
st.write("class 1 is Uninfected")
if uploaded_image is not None: #if image exist
    #open the image
    img = Image.open(uploaded_image)
    #show image using streamlit
    st.image(img, caption = 'Uploaded Image.')

    #convert image to array
    new_image = np.array(img)

    #resize image to 150*150
    new_image = cv2.resize(new_image, (150,150))

    #normalize image
    new_image = new_image.astype('float32') / 255.0

    #reshape image
    new_image = new_image.reshape(1,150,150,3)

    #make prediction
    prediction = Model.predict(new_image)

    predicted_class = np.argmax(prediction, axis = 1)

    st.write(f'Predicted Class :{predicted_class[0]}')