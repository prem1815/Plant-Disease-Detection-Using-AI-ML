import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from keras.preprocessing import image

def discover_directories(root_directory):
    train_dir = os.path.join(root_directory, 'train')
    valid_dir = os.path.join(root_directory, 'valid')
    test_dir = os.path.join(root_directory, 'test')

    return train_dir, valid_dir, test_dir

def load_data(directory):
    images = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(class_name)
    return np.array(images), np.array(labels)

# Define the root directory containing subdirectories for train, valid, and test
root_directory = r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset'

# Automatically discover train, valid, and test directories
train_dir, valid_dir, test_dir = discover_directories(root_directory)

# Define image size
img_size = (224, 224)

# Load training, validation, and test images and labels
X_train, y_train = load_data(train_dir)
X_valid, y_valid = load_data(valid_dir)
X_test, y_test = load_data(test_dir)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
y_test_encoded = label_encoder.transform(y_test)

# Create a K-Nearest Neighbors classifier
k_values = list(range(1, 21))
accuracy_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    # Use cross-validation to estimate the accuracy
    scores = cross_val_score(knn_classifier, X_train.reshape(X_train.shape[0], -1), y_train_encoded, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())

# Plot the Elbow criterion
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Elbow Criterion for KNN')
plt.show()

# Choose the best k based on the Elbow criterion
best_k = k_values[np.argmax(accuracy_scores)]
print(f'Best k based on Elbow criterion: {best_k}')

# Create a K-Nearest Neighbors classifier with the best k
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train_encoded)

# Get predictions on the test set
predicted_classes_test = best_knn_classifier.predict(X_test.reshape(X_test.shape[0], -1))

# Evaluate the model on the test set
accuracy_test = accuracy_score(y_test_encoded, predicted_classes_test)
print(f'Test Accuracy with Best k: {accuracy_test * 100:.2f}%')

# Display classification report for the test set
print("Classification Report:")
print(classification_report(y_test_encoded, predicted_classes_test))

# Create a confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test_encoded, predicted_classes_test)

# Plot a Heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - K-Nearest Neighbors')
plt.show()

# Display some example predictions
random_indices = np.random.choice(len(X_test), size=5, replace=False)
for index in random_indices:
    img = X_test[index] / 255.0  # Normalize the image
    plt.imshow(img)
    true_class = label_encoder.inverse_transform([y_test_encoded[index]])[0]
    predicted_class = label_encoder.inverse_transform([predicted_classes_test[index]])[0]
    plt.title(f"True: {true_class}, Predicted: {predicted_class}")
    plt.show()
