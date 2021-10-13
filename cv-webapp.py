import cv2
import numpy as np
import streamlit as st
import os
from build_averageRGB import build_dataframe_average_rgb
from mosaic_generator import builder

# Keep the state of the button press between actions
@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed": None}

st.title = ("Beginner Computer Vision Projects")

st.markdown("""
# Computer Vision Projects for Beginner
## Web app includes the following features:

1. Edge Detection
2. Photo Sketching
3. Detecting Contours
4. Collage Mosaic Generator
""")
rmseThreshold = 0.5
threshold = 150
lowThreshold = 100
highThreshold = 150
isImg = False
operator = "-"
is_generated = button_states()


st.header("Upload your image")
image_file = st.file_uploader("", type=['png','jpeg','jpg'])
if image_file is not None:
    isImg = True
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(image_file.read()), dtype='uint8')
    img = cv2.imdecode(file_bytes, 1)
    # To see image details
    file_size = {"width":img.shape[0], "height":img.shape[1]}
    st.write("Image size", file_size)
    st.write("**Original Image**")
    st.image(img, channels="BGR")

# Using radio buttons
features = ["-", "Edge detection", "Photo sketching", "Detecting Contours", "Mosaic Generator"]
st.subheader("Choose your operation")
operator = st.radio("", features)

if operator != "-":
    st.header(operator)
if operator != "Mosaic Generator":
    is_generated.update({"pressed": False})

if isImg:
    if operator == "Edge detection":
        lowThreshold = st.slider("Select low threshold", 0, 255, lowThreshold)
        highThreshold = st.slider("Select high threshold", 0, 255, highThreshold)

        edge_img = cv2.Canny(img, lowThreshold, highThreshold)
        st.write("**Edge image**")
        st.image(edge_img)

    if operator == "Photo sketching":
        k_size = 7
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert gray image
        invert_gray = cv2.bitwise_not(gray)
        # Blur gray image
        blur_img = cv2.GaussianBlur(invert_gray, (k_size, k_size), 0)
        # Invert blurred image
        invblur_img = cv2.bitwise_not(blur_img)
        # Sketch Image
        sketch_img = cv2.divide(gray, invblur_img, scale=256.0)
        st.write("**Sketch image**")
        st.image(sketch_img)

    if operator == "Detecting Contours":
        st.subheader("Change threshold to get better result")
        threshold = st.slider("Select threshold", 0, 255, threshold)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold image to binary
        thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        contours_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        st.write("**Image with contours**")
        st.image(contours_img, channels="BGR")

    if operator == "Mosaic Generator":
        st.markdown("""
        ### Please enter folder path for image source to generate mosaic image; input image and save image path respectively.
        """)
        st.write("Source path")
        source_path = st.text_input("Your image source to generate mosaic. Example: E:\\example")
        st.write("Save image path")
        output_path = st.text_input("Path to save your mosaic image. Example: E:\\save_dir\\save.jpg")

        generate = st.button("Generate")

        if generate:
            is_generated.update({"pressed": True})

        if is_generated["pressed"]:
            check_source = os.path.exists(source_path)
            dirname = os.path.dirname(output_path) 
            check_ouput = os.path.exists(dirname)
            st.subheader("Change threshold to get better result")
            rmseThreshold = st.slider("Select threshold", 0.0, 2.0, rmseThreshold, step=0.1)
            if check_source and check_ouput:
                if not os.path.exists('Avg_RGB_dataset.csv'):
                    build_dataframe_average_rgb(source_path)
                
                st.write("---------------------GENERATING--------------------")
                mosaic_img = builder(source_path, img, output_path, rmse_threshold=rmseThreshold)
                st.write("------------------------DONE------------------------")
                st.write("**Mosaic Image**")
                st.image(mosaic_img)
            else:
                st.markdown("<font color='red'>INVALID PATH</font>", unsafe_allow_html=True)

