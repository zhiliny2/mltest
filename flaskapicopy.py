import cv2
import numpy as np
import streamlit as st
# from keras.models import load_model as keras_load_model
from PIL import Image
import tensorflow as tf
import os
import urllib.request
import subprocess

# Function to load the model
# @st.cache(allow_output_mutation=True)
# def load_custom_model():
#     if not os.path.isfile('model.h5'):
#         urllib.request.urlretrieve('https://github.com/zhiliny2/mltest/raw/master/bmi_model_finetuned3.h5', 'model.h5')
#     return tf.keras.models.load_model('model.h5')

if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://github.com/zhiliny2/mltest/raw/master/bmi_model_finetuned3.h5"'], shell=True)

# Load the model
custom_resnet50_model = tf.keras.models.load_model('model.h5', compile=False)

# Load the Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

def predict_bmi(image, model):
    # Preprocess the image
    image = preprocess_image(image)

    # Expand dimensions to match input shape
    image = np.expand_dims(image, axis=0)

    # Perform the prediction
    bmi_prediction = model.predict(image)[0][0]
    return bmi_prediction

# Create a Streamlit app
def main():
    st.title("BMI Estimation from Uploaded Pictures")
    st.text("Upload an image to estimate BMI.")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Convert the image to BGR format for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw bounding boxes around the faces
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region
            face_region = image[y:y + h, x:x + w]

            # Make a BMI prediction for the face region
            bmi_prediction = predict_bmi(face_region, custom_resnet50_model)

            # Add the BMI prediction text to the image
            text = "BMI: {:.2f}".format(bmi_prediction)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the image back to RGB format for display in Streamlit
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image in Streamlit
        st.image(image, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()
