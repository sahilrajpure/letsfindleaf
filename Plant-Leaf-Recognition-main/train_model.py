import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Define the dataset path (Modify this with your dataset folder)
dataset_path = "d:/images"  

# List of species (Ensure they match your dataset folder names)
labels = ["Acer",
          "Alnus incana",
          "Fagus sylvatica",
          "Populus tremula", 
          "Quercus",
          "Salix alba",
          "Salix aurita",
          "Sorbus aucuparia",
          "Sorbus intermedia",
          "Tilia"]

# Encode labels as numbers
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Function to extract features from an image
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to standard size
    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

# Load images and labels
X, y = [], []
for label in os.listdir(dataset_path):
    if label in labels:
        label_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            features = extract_features(img_path)
            X.append(features)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
y_encoded = label_encoder.transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train an SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "leaf_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Model saved as 'leaf_model.pkl'")
