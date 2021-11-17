import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ------- init setting -------
st.set_page_config(
    page_title="Computer Vision HW1 - F14071075",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title('Introduction to Image Processing, Computer Vision and Deep Learing')
st.subheader("Homework 1")
# ------- end -------

# select box:
topic = st.selectbox("Select a topic",
        (
            '(1) Image Prcessing', 
            '(2) Image Smoothing',
            '(3) Edge Detection', 
            '(4) Transforms', 
            '(5) Training Cifar-10 Classifier Using VGG16')
        )

# ------- HW1 - 1 -------
if topic == '(1) Image Prcessing':
    st.header("Image Processing")
    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type='jpg')

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        # show image
        st.image(cv_image_0, channels="BGR")
    
        # Show image size
        st.subheader("Image shape: ", cv_image_0.shape)
        st.write("  Height: ", cv_image_0.shape[0])
        st.write("  Width: ", cv_image_0.shape[1])
        st.write("  Channels: ", cv_image_0.shape[2])
        
        # Show Color Separation
        st.subheader("Color Separation")
        b = cv_image_0.copy()
        b[:,:,1], b[:,:,2] = 0, 0
        g = cv_image_0.copy()
        g[:,:,0], g[:,:,2] = 0, 0
        r = cv_image_0.copy()
        r[:,:,0], r[:,:,1] = 0, 0
        st.image(b, channels="BGR")
        st.image(g, channels="BGR")
        st.image(r, channels="BGR")

        # Show Color Transformation
        st.subheader("Color Transformation")
        gray = cv2.cvtColor(cv_image_0, cv2.COLOR_BGR2GRAY)
        gray.shape
        st.image(gray)
        _b, _g, _r = cv2.split(cv_image_0)
        gray2 = np.floor_divide( (_b + _g + _r), 3)
        st.image( gray2 )

        # Show Blending
        st.subheader("Blending")
        image_1 = st.file_uploader("Upload 1 of 2 Image to combine", type='jpg')
        image_2 = st.file_uploader("Upload 2 of 2 Image to combine", type='jpg')
        weight = st.slider("Weight:", 0, 100) / 100

        if (image_1 is not None) and (image_2 is not None):
            cv_image_1 = cv2.imdecode(np.asarray(bytearray(image_1.read()), dtype=np.uint8), 1)
            cv_image_2 = cv2.imdecode(np.asarray(bytearray(image_2.read()), dtype=np.uint8), 1)
            cv_image_cat = np.concatenate((
                    cv_image_1, cv_image_2
                ) , axis=1)
            st.image(cv_image_cat, channels="BGR")
            dest = cv2.addWeighted( cv_image_1, weight, cv_image_2, 1-weight, 0)
            st.image(dest, channels="BGR")
    # ------- end -------
# ------- end -------

# ------- HW1 - 2 -------
if topic == '(2) Image Smoothing':
    st.header("Image Smoothing")
    
    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type='jpg')

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        # show image
        st.image(cv_image_0, channels="BGR")
        
        # Show Gaussian Blur
        st.subheader("Gaussian Blur:")
        dest = cv2.GaussianBlur(cv_image_0, (5,5), cv2.BORDER_DEFAULT )
        st.image(dest, channels="BGR")
        
        # Show Bilateral Filtering
        st.subheader("Bilateral Filtering:")
        dest = cv2.bilateralFilter(cv_image_0, 90, 9, 9)
        st.image(dest, channels="BGR")

        # Show Median Blur
        st.subheader("Median Blur (3x3):")
        dest = cv2.medianBlur(cv_image_0, 3)
        st.image(dest, channels="BGR")
        st.subheader("Median Blur (5x5):")
        dest = cv2.medianBlur(cv_image_0, 5)
        st.image(dest, channels="BGR")
    # ------- end -------

# ------- end -------

# ------- HW1 - 3 -------
if topic == '(3) Edge Detection':
    st.header("Edge Detection")

    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type='jpg')

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        # show image
        st.image(cv_image_0, channels="BGR")
    # ------- end -------

# ------- end -------

# ------- HW1 - 4 -------
if topic == '(4) Transforms': 
    st.header("Transforms")

    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type='jpg')

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        # show image
        st.image(cv_image_0, channels="BGR")
    # ------- end -------

# ------- end -------
# ------- end -------

# ------- HW1 - 5 -------
if topic == '(5) Training Cifar-10 Classifier Using VGG16':
    st.header("Training Cifar-10 Classifier Using VGG16")

    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type='jpg')

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        # show image
        st.image(cv_image_0, channels="BGR")
    # ------- end -------

# ------- end -------
# ------- end -------

