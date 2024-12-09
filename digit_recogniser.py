import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle 

def load_model():
    url = "https://raw.githubusercontent.com/Yeatrix/DigitRecogniser/main/knn_model.pkl"  
    response = requests.get(url)
    response.raise_for_status()  
    model = pickle.loads(response.content)
    return model

knn_model = load_model()

st.title('Digit Recognizer')
st.markdown('''
Try to write a digit!
''')


SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')


if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="Your Drawing", use_column_width=False)

    def preprocess_image(image):
        image = Image.fromarray((image[:, :, :3] * 255).astype(np.uint8))  
        image = ImageOps.grayscale(image) 
        image = image.resize((28, 28)) 
        image = np.array(image) / 255.0 
        return image.flatten().reshape(1, -1)  

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            input_data = preprocess_image(canvas_result.image_data)
            prediction = knn_model.predict(input_data)
            st.write(f"Predicted Digit: {prediction[0]}")
