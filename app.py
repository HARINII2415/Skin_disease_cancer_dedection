import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')


# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)  # Resize image
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    img = img / 255.0  # Normalize pixel values
    return img

# Define the Streamlit app
def app():
    st.title('🩺 Skin Cancer Detection App')

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is not None:
            # Display the image
            st.image(img, caption='Uploaded Image', use_column_width=True)
            
            # Preprocess the image
            img = preprocess_image(img)

            # Make a prediction
            pred = model.predict(img)
            pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
            pred_prob = pred[0][0]
            
            # Show the prediction result
            st.subheader(f'Prediction: **{pred_label}**')
            st.write(f'🩸 Probability Of Skin Cancer: **{pred_prob:.2f}**')
        else:
            st.error("Error loading the image. Please try another file.")

# Run the app
if __name__ == '__main__':
    app()
