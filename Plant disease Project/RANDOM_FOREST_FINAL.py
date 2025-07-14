import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

def discover_directories(root_directory):
    train_dir = os.path.join(root_directory, 'train')
    valid_dir = os.path.join(root_directory, 'valid')
    test_dir = os.path.join(root_directory, 'test')

    return train_dir, valid_dir, test_dir

# Define the root directory containing subdirectories for train, valid, and test
root_directory = r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset'

# Automatically discover train, valid, and test directories
train_dir, valid_dir, test_dir = discover_directories(root_directory)

# Define image size
img_size = (224, 224)

# Function to load images and convert them to numpy arrays
def load_images(directory):
    images = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load training, validation, and test images
X_train, y_train = load_images(train_dir)
X_valid, y_valid = load_images(valid_dir)
X_test, y_test = load_images(test_dir)

# Flatten the image arrays
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
print("Training Random Forest...")
rf_classifier.fit(X_train_flat, y_train)
print("Training complete.")

# Get predictions on the validation set
predicted_classes_valid = rf_classifier.predict(X_valid_flat)

# Evaluate the model on the validation set
accuracy_valid = accuracy_score(y_valid, predicted_classes_valid)
print(f'Validation Accuracy: {accuracy_valid * 100:.2f}%')

# Get predictions on the test set
predicted_classes_test = rf_classifier.predict(X_test_flat)

# Evaluate the model on the test set
accuracy_test = accuracy_score(y_test, predicted_classes_test)
print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# Display classification report for test set
print("Classification Report:")
print(classification_report(y_test, predicted_classes_test))

# Create a confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, predicted_classes_test)

# Plot a Heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['healthy', 'lateblight', 'earlyblight'],
            yticklabels=['healthy', 'lateblight', 'earlyblight'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.show()
