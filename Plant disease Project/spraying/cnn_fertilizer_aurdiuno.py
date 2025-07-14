'''
const int motorPin = 9; // Pin connected to the pumping motor
unsigned long startTime; // Variable to store the start time
unsigned long duration; // Duration in milliseconds

void setup() {
  Serial.begin(9600); // Start serial communication at 9600 baud rate
  pinMode(motorPin, OUTPUT); // Set motor pin as output
}

void loop() {
  if (Serial.available() > 0) { // Check if data is available to read
    char receivedChar = Serial.read(); // Read the incoming byte

    // Check the received command 
    if (receivedChar == '1') {
      // Start the motor and record the start time
      digitalWrite(motorPin, HIGH);
      startTime = millis();
    }
  }

  // Check if motor is running and the elapsed time exceeds the set duration
  if (digitalRead(motorPin) == HIGH && millis() - startTime >= duration) {
    // Stop the motor
    digitalWrite(motorPin, LOW);
  }
}


'''
import serial
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import time

# Establish serial connection with Arduino
ser = serial.Serial('COM9', 9600) 

# Load the trained machine learning model
model = load_model(r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\spraying\spray_model.h5')

# Function to predict the class of an image and control fertilizer spraying
def predict_and_control_fertilizer(image_path, ser, model):
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

        top_class, _ = sorted_probabilities[0]

        # Display prediction result
        print('Predicted Class:', top_class)

        # Prompt user whether to activate the sprayer
        user_input = input("Do you want to activate the sprayer? (yes/no): ")

        # Send command to Arduino based on user input and predicted class
        if user_input.lower() == 'yes':
            ser.write(b'1')  # Command to start spraying fertilizer
            print("Sprayer activated for 10 seconds.")
            time.sleep(10)  # Delay for 10 seconds
            ser.write(b'0')  # Command to stop spraying fertilizer
            print("Sprayer deactivated.")
        else:
            ser.write(b'0')  # Command to stop spraying fertilizer
            print("Sprayer deactivated.")

    except Exception as e:
        print(f"Error: {str(e)}")
        # Send command to Arduino to stop spraying fertilizer
        ser.write(b'0')

# Example usage
user_image_path = input("Enter the path of the image you want to check: ")
predict_and_control_fertilizer(user_image_path, ser, model)

# Close the serial connection
ser.close()

