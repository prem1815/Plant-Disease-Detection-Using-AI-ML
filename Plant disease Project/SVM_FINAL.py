import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

# Function to extract features from an image
def extract_features(image):
    # Local Binary Pattern (LBP)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # Color Histograms
    hist_color = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_color = hist_color.flatten()
    hist_color /= (hist_color.sum() + 1e-7)

    # Hu Moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    # Combine features
    features = np.concatenate([hist_lbp, hist_color, hu_moments])

    return features

# Specify the dataset directory structure
dataset_path = r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset'

# Get class names from the 'train' directory
train_dir = os.path.join(dataset_path, 'train')
classes = os.listdir(train_dir)

X, y = [], []

# Loop through each class in the 'train' directory
for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    
    # Loop through each image in the class
    for file_name in os.listdir(class_path):
        image_path = os.path.join(class_path, file_name)
        img = cv2.imread(image_path)
        features = extract_features(img)
        X.append(features)
        y.append(class_name)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Support Vector Machine (SVM) classifier with grid search
param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf', 'poly']}
svm_classifier = GridSearchCV(make_pipeline(SVC()), param_grid, cv=5)
svm_classifier.fit(X_train_scaled, y_train_encoded)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report with zero_division parameter
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=classes, zero_division=1))

# Plotting the confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plotting ROC curve (only for binary classification)
if len(classes) == 2:
    fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
