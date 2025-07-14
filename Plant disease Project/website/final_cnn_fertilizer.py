# For potato leaf disease prediction
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('Potato Leaf Disease Prediction and Fertilizer Suggestion')
st.write("Welcome to our website! Explore the latest information and predictions about potato leaf diseases. Upload an image of a potato leaf, and our model will provide insights into potential diseases, along with confidence levels. Additionally, get suggestions for the appropriate fertilizer based on the leaf condition. Stay informed and take proactive measures to ensure the health of your potato plants.")
st.write("NOTE: This platform predicts leaf characteristics, but occasional inaccuracies may occur as it's in the early stages of development. Regarding fertilizer recommendations, the website provides approximate predicted amounts. It's advisable to verify the predicted data before applying any fertilizers!")

def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)

        # Disease and Fertilizer Prediction
        result, confidence, fertilizer_suggestion, top_class, fertilizer_percentage = predict(image)
        st.write('Disease Prediction: {}'.format(result))
        st.write('Confidence: {}%'.format(confidence))
        st.write('Fertilizer Suggestion: {}'.format(fertilizer_suggestion))
        st.write('Top Class: {}'.format(top_class))
        st.write('Approximate Fertilizer Percentage: {}%'.format(fertilizer_percentage))

def predict(image):
    with st.spinner('Loading Model...'):
        model = keras.models.load_model('new_model.h5', compile=False)

    shape = ((256, 256, 3))
    model = keras.Sequential([hub.KerasLayer(model, input_shape=shape)])     
    test_image = image.resize((256, 256))
    test_image_array = keras.preprocessing.image.img_to_array(test_image)
    test_image_array /= 255.0
    test_image_array = np.expand_dims(test_image_array, axis=0)

    class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    # Disease Prediction
    disease_prediction = model.predict(test_image_array)
    disease_confidence = round(100 * np.max(disease_prediction[0]), 2)
    disease_result = class_name[np.argmax(disease_prediction)]

    # Fertilizer Prediction
    class_labels = ['Low Nitrogen', 'High Nitrogen', 'Balanced']
    fertilizer_predictions = disease_prediction  # Use the same prediction for fertilizer (replace with actual logic if different)
    class_probabilities = dict(zip(class_labels, fertilizer_predictions.flatten()))
    sorted_probabilities = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

    top_class, top_probability = sorted_probabilities[0]

    if top_class == 'Low Nitrogen':
        fertilizer_suggestion = "Use a Low Nitrogen Fertilizer"
    elif top_class == 'Balanced':
        fertilizer_suggestion = "Use a Balanced Fertilizer"
    elif top_class == 'High Nitrogen':
        fertilizer_suggestion = "Use a High Nitrogen Fertilizer"
    else:
        fertilizer_suggestion = "Unable to suggest fertilizer"

    # Calculate percentage based on the top probability
    fertilizer_percentage = round(top_probability * 100, 2)

    return disease_result, disease_confidence, fertilizer_suggestion, top_class, fertilizer_percentage

if __name__ == '__main__':
    main()
