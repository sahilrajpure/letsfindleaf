import streamlit as st
import numpy as np
import cv2
import joblib
import wikipedia
import matplotlib.pyplot as plt
import threading
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
import subprocess
import webbrowser
import time


# Load trained model, label encoder, and accuracy score (if available)
model = joblib.load("Plant-Leaf-Recognition-main/leaf_model.pkl")
label_encoder = joblib.load("Plant-Leaf-Recognition-main/label_encoder.pkl")
try:
    model_accuracy = joblib.load("model_accuracy.pkl")  # Assuming you saved accuracy during training
except:
    model_accuracy = None  # Handle case where accuracy file is missing

# Fetch plant info from Wikipedia
def get_plant_info(species_name):
    try:
        summary = wikipedia.summary(species_name + " plant", sentences=3)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"⚠️ Multiple matches found: {e.options[:5]}..."
    except wikipedia.exceptions.PageError:
        return "🌱 No detailed information found on Wikipedia."

# Extract features from an image
def extract_features(img_array):
    img_gray = rgb2gray(img_array)
    img_resized = cv2.resize(img_gray, (128, 128))
    features = hog(img_resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features.reshape(1, -1), features

# Apply mask to remove green background
def apply_mask(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    range1 = (36, 0, 0)
    range2 = (86, 255, 255)
    mask = cv2.inRange(hsv, range1, range2)
    
    result = img.copy()
    result[mask == 0] = (255, 255, 255)  # Convert masked areas to white
    return result

# Convert image to grayscale
def convert_to_grayscale(img_array):
    return rgb2gray(img_array)

def main():
    # Set page configuration
    st.set_page_config(page_title="Leaf Classifier", layout="wide")

    # Display the college header image
    st.image("Plant-Leaf-Recognition-main/logoheade.png", use_container_width=True)

    # Button to open the external plant analyzer
    st.markdown(
        """
        <a href="https://mvluplantfilter.streamlit.app/?embed_options=light_theme,show_padding" target="_blank">
            <button style="
                background-color:#2E7D32; 
                color:white; 
                font-size:16px; 
                border-radius:8px; 
                padding:8px 16px; 
                border:none;
                cursor:pointer;">
                🌱 Similar Plant Analyzer with AI
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body, .stApp {
            background: #EAF4D3; /* Light Green */
            font-family: 'Poppins', sans-serif;
            color: #1B5E20; /* Dark Forest Green */
        }

        h1, h2, h3 {
            text-align: center;
            font-weight: 600;
            color: #2E7D32; /* Dark Green */
        }

        .stButton>button {
            background-color: #2E7D32; /* Dark Green */
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 16px;
            transition: 0.3s ease-in-out;
            border: none;
        }

        .stButton>button:hover {
            background-color: #1B5E20; /* Even Darker Green */
            transform: scale(1.05);
        }

        .stMarkdown {{
                background: rgba(255, 255, 255, 0.7);
                padding: 15px;
                border-radius: 10px;
                color: black;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Rest of your Streamlit app code...


    # App Title and Project Overview
    st.title("🌿 Leaf Classifier")
    st.markdown("<h3>Upload a leaf image to classify its species using AI!</h3>", unsafe_allow_html=True)
    st.markdown("""
    ## Project Overview
    This **Leaf Classifier** uses Machine Learning to classify plant species based on images of leaves. The model is trained using Histogram of Oriented Gradients (HOG) features to identify the species of a leaf from uploaded images.

    **Features:**
    - Upload an image of a leaf to classify.
    - Get detailed plant information from Wikipedia.
    - Analyze feature distributions and visualize HOG features.

    This app is designed to help you identify plants from their leaves with the power of AI and image processing. The classification model was trained on various leaf images to provide accurate results.

    ### Instructions:
    - Upload a **leaf image** (JPG, PNG, TIFF).
    - Click **Predict** to classify.
    - If confidence is low, a mask is applied automatically for better results.
    """, unsafe_allow_html=True)

    # Sidebar with Instructions
    with st.sidebar:
        st.header("📌 Instructions")
        st.write("""
        1. Upload a **leaf image** (JPG, PNG, TIFF).
        2. Click **Predict** to classify the species.
        3. **Low confidence?** A mask is applied automatically.
        """)
        st.info("Model uses HOG features for classification.")
        

    # File uploader
    image_file = st.file_uploader("Upload a leaf image (JPG, PNG, TIFF)...", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if image_file:
        # Load and display the image
        img = Image.open(image_file)
        img = img.convert("RGB")
        img_array = np.array(img)

if image_file:
    # Load and display the image
    img = Image.open(image_file)
    img = img.convert("RGB")
    img_array = np.array(img)
        
        # Convert to grayscale
        img_gray = convert_to_grayscale(img_array)

        # Apply mask
        img_masked = apply_mask(img_array)

        # Display Images (Original, Grayscale, and Masked)
        st.subheader("📷 Image Processing Stages")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img, caption="📸 Original Image", use_container_width=True)
        with col2:
            st.image(img_gray, caption="⚫ Grayscale Image", use_container_width=True, clamp=True)
        with col3:
            st.image(img_masked, caption="🖼️ Masked Image", use_container_width=True)

        st.write("🔍 **Extracting features and classifying...**")

        # Extract features and predict with probabilities
        features, raw_features = extract_features(img_array)
        probabilities = model.predict_proba(features)[0]  # Get probability distribution
        predicted_index = np.argmax(probabilities)  # Get the index of the highest probability
        confidence_score = probabilities[predicted_index] * 100  # Convert to percentage
        result = model.classes_[predicted_index]  # Get predicted class

        # Apply mask if confidence is low
        if confidence_score < 50:
            st.warning("⚠️ Low confidence detected! Applying a custom mask for better results.")
            features, raw_features = extract_features(img_masked)
            probabilities = model.predict_proba(features)[0]
            predicted_index = np.argmax(probabilities)
            confidence_score = probabilities[predicted_index] * 100
            result = model.classes_[predicted_index]

        species_name = label_encoder.inverse_transform([result])[0]
        st.success(f"🌱 This leaf is from the species: **{species_name}**")

        # Display output accuracy (classification confidence)
        st.success(f"📊 Classification Confidence: **{confidence_score:.2f}%**")

        st.write(f"[🔎 Click here to learn more!](https://www.google.com/search?q={species_name.replace(' ', '+')}+leaf)")

        # Fetch plant info from Wikipedia
        st.subheader("📖 About this Plant")
        plant_info = get_plant_info(species_name)
        st.markdown(f"📝 **{species_name}**: {plant_info}", unsafe_allow_html=True)


        # Visualization Section
        st.subheader("📊 Feature & HOG Analysis")

        # Plot 1: HOG Feature Histogram
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.hist(raw_features, bins=30, color="green", alpha=0.7)
        ax1.set_title("HOG Feature Distribution")
        ax1.set_xlabel("Feature Value")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        # Plot 2: Scatter Plot of HOG Features
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        x_values = np.arange(len(raw_features))
        ax2.scatter(x_values, raw_features, color="blue", alpha=0.6, s=10)
        ax2.set_title("HOG Feature Scatter Plot")
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Feature Value")
        st.pyplot(fig2)

         # Footer
        st.markdown("""
        ---
        🔬 **Built with Python, OpenCV, Scikit-Image, and Streamlit**  
        💡 **Developed by Sahil Rajpure for Plant Enthusiasts & Researchers**   
    """)

if __name__ == '__main__':
    main()
