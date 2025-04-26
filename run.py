import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2

# Define color map for visualization
COLOR_MAP = [
    [0, 0, 0],  # black: sea water
    [0, 255, 255],  # cyan: very high concentrated oil
    [255, 0, 0],  # red: oil spill low concentrated
    [153, 76, 0],  # brown: marine debris
    [0, 153, 0],  # green: rough surface / solid (ship/ice)
]

# Corresponding class names for the colors in the COLOR_MAP
CLASS_NAMES = [
    "Sea Water", 
    "Very High Concentrated Oil", 
    "Oil Spill (Low Concentrated)", 
    "Marine Debris", 
    "Rough Surface / Solid (Ship/Ice)"
]

# Load the pre-trained model
model = load_model(r'./unet_model.h5')

# Streamlit UI setup
st.title("Oil Spill Detection")
st.write("Upload an image to detect oil spills and other elements.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def display_class_name(x, y, mask, class_names):
    """Display class name at the clicked point."""
    class_index = mask[y, x]  # Get the class index based on the clicked pixel
    class_name = class_names[class_index]  # Retrieve the class name
    st.write(f"Class at this point: {class_name}")  # Display class name in Streamlit

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image to match model input
    img = img.resize((256, 256))  # Resize if necessary
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(img_array)

    # Convert prediction to class labels
    predicted_mask = np.argmax(prediction, axis=-1)[0, :, :]  # Get the predicted mask
    predicted_mask_colored = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

    # Map the prediction to the color map
    for i in range(len(COLOR_MAP)):
        predicted_mask_colored[predicted_mask == i] = COLOR_MAP[i]

    # Display the result
    st.subheader("Predicted Mask")
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    
    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask_colored)
    plt.title("Predicted Mask")
    
    # Show the figure with original and predicted mask
    st.pyplot(plt)
    
    # Overlay Mask on Original Image with 50% opacity
    st.subheader("Overlay (Image + Predicted Mask) with 50% Opacity")
    
    # Convert images to float to ensure proper blending
    img_array = np.array(img).astype(np.float32)
    predicted_mask_colored = predicted_mask_colored.astype(np.float32)

    # Blend the images with 50% opacity
    overlay = cv2.addWeighted(img_array, 0.5, predicted_mask_colored, 0.5, 0)

    # Use Plotly to show the overlay with clickable areas
    fig = go.Figure()

    # Add the image with the overlay
    fig.add_trace(go.Image(z=overlay.astype(np.uint8)))

    # Add click event to capture the coordinates
    fig.update_layout(
        clickmode='event+select',
        title="Overlay (Image + Predicted Mask) with 50% Opacity"
    )

    # Handle click events in the figure
    fig.data[0].on_click(lambda trace, points, selector: display_class_name(
        points.point_inds[0] % overlay.shape[1], points.point_inds[0] // overlay.shape[1], 
        predicted_mask, CLASS_NAMES))

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

        # Detect click on the image and display the corresponding class name
    click_x = st.number_input("Click X-coordinate", min_value=0, max_value=overlay.shape[1] - 1, value=0)
    click_y = st.number_input("Click Y-coordinate", min_value=0, max_value=overlay.shape[0] - 1, value=0)

    # Display the class name of the clicked coordinates
    if st.button("Get Class Name"):
        display_class_name(click_x, click_y, predicted_mask, CLASS_NAMES)

