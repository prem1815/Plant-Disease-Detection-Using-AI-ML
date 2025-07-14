from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('leaf_model.h5')

# Function to predict and suggest fertilizer
def predict_and_suggest_fertilizer(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("Image not loaded properly.")

        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        class_labels = ['Low Nitrogen', 'Balanced', 'High Nitrogen']
        class_probabilities = dict(zip(class_labels, predictions.flatten()))
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

        return fertilizer_suggestion, top_class, fertilizer_percentage

    except Exception as e:
        return "Error", None, None

# Set up file upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for image processing
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if the post request has the file part
        if 'image_upload' not in request.files:
            return render_template('error.html', error_message="No file part")

        file = request.files['image_upload']

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('error.html', error_message="No selected file")

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform prediction on the uploaded image
            suggested_fertilizer, top_class, fertilizer_percentage = predict_and_suggest_fertilizer(filepath)

            return render_template('result.html', top_class=top_class, fertilizer_percentage=fertilizer_percentage,
                                    suggested_fertilizer=suggested_fertilizer)

        else:
            return render_template('error.html', error_message="Invalid file format. Allowed formats: png, jpg, jpeg, gif")

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
