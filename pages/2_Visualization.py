import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import plotly.graph_objects as go
import random
import webcolors
from sklearn.cluster import KMeans
import mahotas.features.texture as texture
from sklearn.decomposition import PCA
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


st.markdown("""
    <style>
        body {
            background-image: url("https://cff2.earth.com/uploads/2021/03/16173225/shutterstock_17212516244-scaled.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

#st.title("Plant Disease Classification Dashboard")
st.markdown("<h1 style='text-align: center; font-size: 80px; font-weight: bold;'>Plant Disease Classification Dashboard</h1>", unsafe_allow_html=True)


# Specify the path to the dataset folder
dataset_path = "pages/sample-data/train" 
# Get the list of class folders
class_folders = sorted(os.listdir(dataset_path))





# Sidebar menu options
sidebar_options = ["Image Visualization","Object Detection/Segmentation","Deep Learning Feature Extraction","Color Analysis", "Texture Analysis", "Image Statistics","Class Distribution"]

# Create sidebar
selected_option = st.sidebar.selectbox("Select EDA Task", sidebar_options)



import streamlit as st


def main():
    st.title("Data Description: Plant Disease Dataset")

    st.markdown('<span style="color:#210F76;font-weight:bold;">The Plant Disease Dataset is a comprehensive collection of images depicting various plant diseases affecting different crop species.</span>', unsafe_allow_html=True)
    st.markdown('<span style="color:#210F76;">This dataset consists of about 140K+ images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is created by combining several individual datasets related to plant diseases, providing a wide range of examples for analysis and research purposes.</span>', unsafe_allow_html=True)

    st.markdown("## Images")
    st.markdown('<span style="color:#210F76;">The dataset comprises a large number of images, amounting to several thousand, showcasing different plant species afflicted with various diseases. These images are in the form of JPG files and have varying resolutions.</span>', unsafe_allow_html=True)

    st.markdown("## Plant Species")
    st.markdown('<span style="color:#210F76;">The dataset covers numerous plant species, including crops like apple, blueberry, cherry, corn, grape, peach, potato, raspberry, soybean, strawberry, and tomato, among others. This diversity allows researchers and enthusiasts to study and analyze the diseases prevalent in a broad range of plants.</span>', unsafe_allow_html=True)

    st.markdown("## Disease Labels")
    st.markdown('<span style="color:#210F76;">Each image in the dataset is associated with a specific disease label, indicating the type of disease present in the plant. The dataset includes a comprehensive set of disease classes such as apple scab, apple rust, grape black rot, tomato bacterial spot, tomato late blight, potato early blight, and many others.</span>', unsafe_allow_html=True)

    st.markdown("## Image Organization")
    st.markdown('<span style="color:#210F76;">The dataset is structured into separate folders, with each folder representing a specific plant species. Within each species folder, the images are further organized into subfolders based on the respective disease classes. This organization simplifies navigation and access to specific images of interest.</span>', unsafe_allow_html=True)

    st.markdown("## Dataset Size and Variability")
    st.markdown('<span style="color:#210F76;">The dataset contains a substantial number of images for each plant species and disease class, ensuring a diverse representation of diseases across different plants. This variability contributes to the robustness and generalizability of models trained on the dataset.</span>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()














# Perform actions based on selected option
if selected_option == "Image Statistics":
    # Perform image statistics analysis
    # ...
    st.title("Performing Image Statistics analysis...")
    st.markdown("""
    <style>
        body {
            background-image: url("https://www.latrobe.edu.au/news/announcements/2020/understanding-plant-growth/shutterstock_120646195.jpg/large.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.5);
        }
    </style>
""", unsafe_allow_html=True)
        # Perform image statistics analysis
    image_count = 0
    total_width = 0
    total_height = 0

    # Iterate over the class folders
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)

        # Iterate over the image files in the class folder
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)

            # Update statistics
            image_count += 1
            total_width += image.shape[1]
            total_height += image.shape[0]

    # Calculate average image size
    average_width = total_width // image_count
    average_height = total_height // image_count

    # Display image statistics
    st.write("Total Images:", image_count)
    st.write("Average Image Size:", average_width, "x", average_height)
        # Image size distribution
    image_widths = []
    image_heights = []

    # Image channels
    channel_counts = []

    # Image resolution
    resolutions_widths = []
    resolutions_heights = []

    # Iterate over the class folders
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)

        # Iterate over the image files in the class folder
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)

            # Image size distribution
            image_widths.append(image.shape[1])
            image_heights.append(image.shape[0])

            # Image channels
            channel_counts.append(image.shape[2])

            # Image resolution
            dpi = 96  # Set the DPI value for resolution calculation
            resolution_width = image.shape[1] / dpi
            resolution_height = image.shape[0] / dpi
            resolutions_widths.append(resolution_width)
            resolutions_heights.append(resolution_height)

    # Convert the lists to NumPy arrays
    image_widths = np.array(image_widths)
    image_heights = np.array(image_heights)
    channel_counts = np.array(channel_counts)
    resolutions_widths = np.array(resolutions_widths)
    resolutions_heights = np.array(resolutions_heights)

    # Display image size distribution
    st.subheader("Image Size Distribution")
    fig_size_dist = go.Figure()
    fig_size_dist.add_trace(go.Histogram(x=image_widths, opacity=0.7, name='Width'))
    fig_size_dist.add_trace(go.Histogram(x=image_heights, opacity=0.7, name='Height'))
    fig_size_dist.update_layout(barmode='overlay', xaxis_title='Size', yaxis_title='Frequency')
    st.plotly_chart(fig_size_dist)

    # Display image channels
    st.subheader("Image Channels")
    fig_channels = go.Figure()
    fig_channels.add_trace(go.Scatter(x=np.unique(channel_counts), y=np.bincount(channel_counts)))
    fig_channels.update_layout(xaxis_title='Channels', yaxis_title='Frequency')
    st.plotly_chart(fig_channels)

    # Display image resolution
    st.subheader("Image Resolution")
    fig_resolution = go.Figure()
    fig_resolution.add_trace(go.Scatter(x=resolutions_widths, y=resolutions_heights, mode='markers', marker=dict(color='purple', opacity=0.7)))
    fig_resolution.update_layout(xaxis_title='Width', yaxis_title='Height')
    st.plotly_chart(fig_resolution)




    # ...

elif selected_option == "Image Visualization":
    st.markdown("""
    <style>
        body {
            background-image: url("https://wallpaperaccess.com/full/452682.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.5);
        }
    </style>
""", unsafe_allow_html=True)








    # Perform image visualization
    # ...
    st.title("Performing Image Visualization...")
    # ...
        # Image Previews
    st.subheader("Image Previews")
    random_class = random.choice(class_folders)
    class_path = os.path.join(dataset_path, random_class)
    image_files = os.listdir(class_path)
    random.shuffle(image_files)
    image_preview_cols = st.columns(4)
    for i, image_file in enumerate(image_files[:16]):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path)
        image_preview_cols[i % 4].image(image, use_column_width=True)

    

    # Image Thumbnails with Labels
    st.subheader("Image Thumbnails")
    thumbnails_cols = st.columns(4)
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)
        random.shuffle(image_files)
        for i, image_file in enumerate(image_files[:4]):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            thumbnails_cols[i].image(image, caption=class_folder, use_column_width=True)

        # Image Similarity (Random Images)
    st.subheader("Image Similarity")
    num_cols = min(4, len(class_folders))
    similar_images_cols = st.columns(num_cols)
    for i, class_folder in enumerate(class_folders[:num_cols]):
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)
        random_image_file = random.choice(image_files)
        image_path = os.path.join(class_path, random_image_file)
        image = cv2.imread(image_path)
        similar_images_cols[i].image(image, caption=class_folder, use_column_width=True)










elif selected_option == "Class Distribution":
    # Perform class distribution analysis
    # ...
    st.title("Performing Class Distribution analysis...")
    # ...
    # Class Distribution
    st.subheader("Class Distribution")

    # Count the number of images in each class
    class_counts = []
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        image_files = os.listdir(class_path)
        class_count = len(image_files)
        class_counts.append(class_count)

    # Create a bar chart to display class distribution
    fig = go.Figure(data=go.Bar(x=class_folders, y=class_counts))
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Number of Images",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig)










elif selected_option == "Color Analysis":
    # Perform color analysis
    # ...
    st.write("Performing Color Analysis...")
    # ...
    st.markdown("""
    <style>
        body {
            background-image: url("https://www.wallpapers13.com/wp-content/uploads/2016/04/Light-pink-flowers-Desktop-Wallpaper-full-screen-1280x960.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

        # Color Analysis
    st.subheader("Color Analysis")

    # Select a random image for color analysis
    class_folder = st.selectbox("Select a class", class_folders, key="color_analysis_class")
    class_path = os.path.join(dataset_path, class_folder)
    image_files = os.listdir(class_path)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(class_path, random_image_file)
    image = cv2.imread(image_path)

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate color histograms for each channel
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    # Create a plot to display color histograms
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    ax[0].plot(hist_r, color='r')
    ax[0].set_ylabel('Red Channel')
    ax[1].plot(hist_g, color='g')
    ax[1].set_ylabel('Green Channel')
    ax[2].plot(hist_b, color='b')
    ax[2].set_ylabel('Blue Channel')
    ax[2].set_xlabel('Pixel Value')
    plt.tight_layout()
    st.pyplot(fig)

    # Color Palette
    # Display color palette


    st.subheader("Color Palette")





    # Reshape the image to 1D array of pixels
    pixels = image_rgb.reshape(-1, 3)

    # Perform k-means clustering to extract dominant colors
    num_colors = 5
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = cluster_centers.round().astype(int)

    # Convert RGB values to hexadecimal color codes
    color_palette = []
    for color in cluster_centers:
        color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
        color_palette.append(color_hex)

    # Display color palette
    for color in color_palette:
        st.markdown(f"<span style='background-color:{color}; display:inline-block; width:50px; height:50px;'></span>",
                    unsafe_allow_html=True)
        


    max_width = 500
    height, width, _ = image_rgb.shape
    if width > max_width:
        resized_width = max_width
        resized_height = int(height * max_width / width)
        image_rgb = cv2.resize(image_rgb, (resized_width, resized_height))






    

    # Display the original image
    st.subheader("Original Image")
    st.image(image_rgb, use_column_width=max_width)
    # Display color dominance
    dominant_colors = [color.split("#")[-1] for color in color_palette]
    dominant_colors_text = ", ".join(dominant_colors)
    st.markdown(f"The dominant colors in the image are: {dominant_colors_text}")

    # Provide insights about dominant colors
    if len(dominant_colors) == 1:
        st.subheader("The image is primarily dominated by a single color.")
    elif len(dominant_colors) == 2:
        st.subheader("The image is characterized by a combination of two dominant colors.")
    else:
        st.subheader("The image exhibits a diverse range of dominant colors.")




elif selected_option == "Texture Analysis":
    # Perform Texture analysis
    # ...
    st.title("Performing Texture analysis...")
    # ...

    # Class Distribution
    st.subheader("Texture Analysis")

    st.markdown("""
    <style>
        body {
            background-image: url("https://w0.peakpx.com/wallpaper/454/273/HD-wallpaper-green-leaf-plant-on-white-background.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

        # Select an image for texture analysis
    class_folder = st.selectbox("Select a class", class_folders)
    class_path = os.path.join(dataset_path, class_folder)
    image_files = os.listdir(class_path)
    random_image_file = random.choice(image_files)
    image_path = os.path.join(class_path, random_image_file)
    selected_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    max_width = 500
    height, width, _ = selected_image.shape
    if width > max_width:
        resized_width = max_width
        resized_height = int(height * max_width / width)
        selected_image = cv2.resize(selected_image, (resized_width, resized_height))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2GRAY)

    # Calculate texture features
    features = texture.haralick(gray_image)
    contrast = features.mean(axis=0)[1]
    energy = features.mean(axis=0)[2]
    homogeneity = features.mean(axis=0)[4]
    entropy = features.mean(axis=0)[8]

    # Display the texture analysis results
    st.subheader("Texture Analysis Results")
    st.write("Contrast:", contrast)
    st.write("Energy:", energy)
    st.write("Homogeneity:", homogeneity)
    st.write("Entropy:", entropy)

    # Display the selected image
    st.subheader("Selected Image")
    st.image(selected_image, caption=class_folder, use_column_width=max_width)













elif selected_option == "Deep Learning Feature Extraction":
    # Perform class Texture analysis
    # ...
    st.title("Deep Learning Feature Extraction..")
    # ...

    st.markdown("""
    <style>
        body {
            background-image: url("https://i.pinimg.com/736x/aa/b1/05/aab1054504197c9c65e60393c23e0cbe.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)
        
    # Load pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=False)

    # Set the path to your dataset
    dataset_path = "pages/sample-data/train"

    # Get the list of classes
    class_folders = os.listdir(dataset_path)

    # Select a class for feature extraction
    class_folder = st.selectbox("Select a class", class_folders)
    class_path = os.path.join(dataset_path, class_folder)

    # Get the list of images in the selected class folder
    image_files = os.listdir(class_path)

    # Select an image for feature extraction
    image_file = st.selectbox("Select an image", image_files)
    image_path = os.path.join(class_path, image_file)

     # Load and display the selected image
    selected_image = Image.open(image_path)
    st.image(selected_image)



    selected_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    preprocessed_image = preprocess_input(selected_image)

    # Extract features using VGG16 model
    features = model.predict(np.expand_dims(preprocessed_image, axis=0)).flatten()

    # Display the feature vector
    st.subheader("Feature Vector")
    st.write(features)

    # Create a bar chart for feature visualization
    fig = go.Figure(data=[go.Bar(x=list(range(len(features))), y=features)])

    fig.update_layout(
        title="Feature Extraction Vector",
        xaxis_title="Feature Index",
        yaxis_title="Feature Value",
        bargap=0.5
    )

    st.plotly_chart(fig)





elif selected_option == "Object Detection/Segmentation":
    # Perform class Texture analysis
    # ...
    st.title("Object Detection/Segmentation..")
    # ...
    
    st.markdown("""
    <style>
        body {
            background-image: url("https://images.pexels.com/photos/1353938/pexels-photo-1353938.jpeg?cs=srgb&dl=pexels-min-an-1353938.jpg&fm=jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)


# Load the pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Set the path to your dataset
    dataset_path = "pages/sample-data/train"

    # Get the list of plant folders
    plant_folders = os.listdir(dataset_path)

    # Select a plant folder
    selected_folder = st.selectbox("Select a plant folder", plant_folders)
    folder_path = os.path.join(dataset_path, selected_folder)

    # Get the list of images in the selected plant folder
    image_files = os.listdir(folder_path)

    # Select an image
    selected_image_file = st.selectbox("Select an image", image_files)
    image_path = os.path.join(folder_path, selected_image_file)

    # Load and display the selected image
    selected_image = Image.open(image_path)
    st.image(selected_image)

    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_image = transform(selected_image)

    # Perform object detection
    with torch.no_grad():
        predictions = model([input_image])

    # Extract the bounding boxes, labels, and scores from the prediction
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # Filter the results for diseased parts (adjust as per your dataset)
    diseased_boxes = boxes[labels == 1]
    diseased_scores = scores[labels == 1]

    # Visualize the detected diseased parts
    if len(diseased_boxes) > 0:
        # Draw bounding boxes on the image
        output_image = np.array(selected_image.copy())
        for box, score in zip(diseased_boxes, diseased_scores):
            cv2.rectangle(output_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(output_image, f"Score: {score:.2f}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the output image
        st.image(output_image, channels="BGR")
    else:
        st.header("No diseased parts detected.")


    # Load pre-trained Faster R-CNN model











# Continue with the other options...

