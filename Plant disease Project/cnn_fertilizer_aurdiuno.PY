# Arduino code for fertilizer sprayer system with a pumping motor and automatic stop after 10 seconds
"""const int motorPin = 9; // Pin connected to the pumping motor
unsigned long startTime; // Variable to store the start time
const unsigned long duration = 10000; // Duration in milliseconds (10 seconds)

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

  // Check if 10 seconds have elapsed since the motor started
  if (digitalRead(motorPin) == HIGH && millis() - startTime >= duration) {
    // Stop the motor
    digitalWrite(motorPin, LOW);
  }
}
"""


import serial
import numpy as np
import cv2

# Establish serial connection with Arduino
ser = serial.Serial('COM3', 9600) # Change 'COM3' to the appropriate port

# Function to predict the class of an image and control fertilizer spraying
def predict_and_control_fertilizer(image_path, ser):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("Image not loaded properly.")

        # Preprocess the image as needed
        # Perform inference with your machine learning model
        # For demonstration, let's assume the ML model predicts a class label '1' to spray fertilizer

        # Send command to Arduino to start spraying fertilizer
        ser.write(b'1')

        # You can also read response from Arduino if needed
        # response = ser.readline().decode('ascii')

    except Exception as e:
        print(f"Error: {str(e)}")
        # Send command to Arduino to stop spraying fertilizer
        ser.write(b'0')

# Example usage
user_image_path = input("Enter the path of the image you want to check: ")
predict_and_control_fertilizer(user_image_path, ser)

