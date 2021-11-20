import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch

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

# Stable Variances:
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = 0
if 'state_2' not in st.session_state:
    st.session_state['state_2'] = 0
if 'state_3' not in st.session_state:
    st.session_state['state_3'] = 0
if 'state_4' not in st.session_state:
    st.session_state['state_4'] = 0
if 'state_5' not in st.session_state:
    st.session_state['state_5'] = 0

# ----- Warning Part -----
st.write("Github: [https://github.com/RoyChao19477/opencv_hw1](https://github.com/RoyChao19477/opencv_hw1)")
st.write("Author: F14071075@2021")
# ----- end -----

# ------- HW1 - 1 -------
if topic == '(1) Image Prcessing':
    st.header("Image Processing")
    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        
        if st.button("Show Image Detail:"):
            st.session_state.state_1 = 1
        if st.session_state.state_1 >= 1:
            st.write("  Shape: ", cv_image_0.shape)
            st.write("  Size: ", cv_image_0.size)
            st.write("  Data Type: ", cv_image_0.dtype)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
            # show image
            st.image(cv_image_0, channels="BGR")
    
        if st.button("Image Shape:"):
            st.session_state.state_1 = 2
        if st.session_state.state_1 >= 2:
            # Show image size
            st.subheader("Image shape: ", cv_image_0.shape)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
            st.write("  Channels: ", cv_image_0.shape[2])
        
        if st.button("Color Separation"):
            st.session_state.state_1 = 3
        if st.session_state.state_1 >= 3:
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

        if st.button("Color Transformation"):
            st.session_state.state_1 = 4
        if st.session_state.state_1 >= 4:
            # Show Color Transformation
            st.subheader("Color Transformation")
            gray = cv2.cvtColor(cv_image_0, cv2.COLOR_BGR2GRAY)
            st.write(gray.shape)
            st.write("Use cvtColor: cv2.COLOR_BGR2GRAY")
            st.image(gray)
            _b, _g, _r = cv2.split(cv_image_0)
            gray2 = (_b * 0.3333 + _g * 0.3333 + _r * 0.3333).astype(int)
            gray3 = (_b * 0.1140 + _g * 0.5870 + _r * 0.2989).astype(int)
            st.write("0.3333 * B + 0.3333 * G + 0.3333 * R :")
            st.image( gray2 )
            st.write("0.1140 * B + 0.5870 * G + 0.2989 * R :")
            st.image( gray3 )


        if st.button("Blending"):
            st.session_state.state_1 = 5
        if st.session_state.state_1 >= 5:
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
    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)
        

        if st.button("Show Image Detail:"):
            st.session_state.state_2 = 1
        if st.session_state.state_2 >= 1:
            st.write("  Shape: ", cv_image_0.shape)
            st.write("  Size: ", cv_image_0.size)
            st.write("  Type: ", cv_image_0.size)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
            # show image
            st.image(cv_image_0, channels="BGR")
        
        if st.button("Gaussian Blur:"):
            st.session_state.state_2 = 2
        if st.session_state.state_2 >= 2:
            # Show Gaussian Blur
            st.subheader("Gaussian Blur:")
            dest = cv2.GaussianBlur(cv_image_0, (5,5), cv2.BORDER_DEFAULT )
            st.image(dest, channels="BGR")
        
        if st.button("Bilateral Filtering:"):
            st.session_state.state_2 = 3
        if st.session_state.state_2 >= 3:
            # Show Bilateral Filtering
            st.subheader("Bilateral Filtering:")
            dest = cv2.bilateralFilter(cv_image_0, 90, 9, 9)
            st.image(dest, channels="BGR")

        if st.button("Median Blur:"):
            st.session_state.state_2 = 4
        if st.session_state.state_2 >= 4:
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
    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        if st.button("Show Image Detail:"):
            st.session_state.state_3 = 1
        if st.session_state.state_3 >= 1:
            st.write("  Shape: ", cv_image_0.shape)
            st.write("  Size: ", cv_image_0.size)
            st.write("  Data Type: ", cv_image_0.dtype)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
            # show image
            st.image(cv_image_0, channels="BGR")
        
        if st.button("After Gray (0.1140B + 0.5870G + 0.2989R):"):
            st.session_state.state_3 = 2
        if st.session_state.state_3 >= 2:
            st.write("Gray")
            _b, _g, _r = cv2.split(cv_image_0)
            gray = (_b * 0.1140 + _g * 0.5870 + _r * 0.2989).astype(int)
            st.image( gray )
        
        if st.button("Gaussian Blur:"):
            st.session_state.state_3 = 3
        if st.session_state.state_3 >= 3:
            g_init = np.array(
                        [
                            [[-1, -1], [0, -1], [1, -1]],
                            [[-1, 0], [0, 0], [1, 0]],
                            [[-1, 1], [0, 1], [1, 1]],
                        ])
            st.write("init G:")
            st.write(g_init)
            
            X2 = g_init[:,:,0]**2
            Y2 = g_init[:,:,1]**2
            PI = 3.141592653589
            VAR = np.var(g_init)
            
            g = ( 1 / (2 * PI * VAR ) * np.exp( -(X2 + Y2) / ( 2 * VAR )))
            g_norm = g / g.sum()
            
            st.write("Normalized G:")
            st.write(g_norm)

            img = np.zeros(
                    (
                        gray.shape[0] - 2,
                        gray.shape[1] - 2
                    )
                    )
        if st.button("After Gaussian Blur:"):
            st.session_state.state_3 = 4
        if st.session_state.state_3 >= 4:
            st.write("After Gaussian Blur with normalized G:")
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i,j] = (gray[ i:i+3, j:j+3 ] * g_norm).sum()
            img = img.astype(int)
            st.image( img )

        if st.button("Sobel X:"):
            st.session_state.state_3 = 5
        if st.session_state.state_3 >= 5:
            st.write("If result < 0 -> 0")
            st.write("If result > 255 -> 255")
            sobel_x = np.array(
                    [
                        [ -1, 0, 1],
                        [ -2, 0, 2],
                        [-1, 0, 1]
                    ]
                    )
            img2 = np.zeros(( gray.shape[0] - 2, gray.shape[1] - 2))
            for i in range(img2.shape[0]):
                for j in range(img2.shape[1]):
                    val = (gray[ i:i+3, j:j+3 ] * sobel_x).sum()
                    if val < 0: 
                        val = 0
                    if val > 255:
                        val = 255
                    img2[i, j] = val
            img2 = img2.astype(int)
            st.image( img2 )

        if st.button("Sobel Y:"):
            st.session_state.state_3 = 6
        if st.session_state.state_3 >= 6:
            st.write("If result < 0 -> 0")
            st.write("If result > 255 -> 255")
            sobel_x = np.array(
                    [
                        [ 1, 2, 1],
                        [ 0, 0, 0],
                        [-1, -2, -1]
                    ]
                    )
            img3 = np.zeros(( gray.shape[0] - 2, gray.shape[1] - 2))
            for i in range(img3.shape[0]):
                for j in range(img3.shape[1]):
                    val2 = (gray[ i:i+3, j:j+3 ] * sobel_x).sum()
                    if val2 < 0: 
                        val2 = 0
                    if val2 > 255:
                        val2 = 255
                    img3[i, j] = val2
            img3 = img3.astype(int)
            st.image( img3 )

        if st.button("Sobel X with Sobel Y (Megnitude):"):
            st.session_state.state_3 = 7
        if st.session_state.state_3 >= 7:
            img4 = np.sqrt( (img2**2 + img3**2) )
            img4 *= ( 255 / img4.max() )
            img4 = img4.astype(int)
            st.image( img4 )
    # ------- end -------

# ------- end -------

# ------- HW1 - 4 -------
if topic == '(4) Transforms': 
    st.header("Transforms")

    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        if st.button("Show Image Detail:"):
            st.session_state.state_4 = 1
        if st.session_state.state_4 >= 1:
            st.write("  Shape: ", cv_image_0.shape)
            st.write("  Size: ", cv_image_0.size)
            st.write("  Data Type: ", cv_image_0.dtype)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
            # show image
            st.write("Image shape (before): ", cv_image_0.shape)
            st.image(cv_image_0, channels="BGR")
        
        if st.button("Reshape:"):
            st.session_state.state_4 = 2
        if st.session_state.state_4 >= 2:
            # show resize image
            st.subheader("Reshape")
            cv_image_resize = cv2.resize(cv_image_0, (256,256))
            st.write("Image shape (after resize): ", cv_image_resize.shape)
            st.image(cv_image_resize, channels="BGR")
        
        if st.button("Translate:"):
            st.session_state.state_4 = 3
        if st.session_state.state_4 >= 3:
            # show translation
            st.subheader("Translation")
            M = cv2.getAffineTransform(
                    np.float32([[0,0],  [128,128], [255, 0]]), 
                    np.float32([[0,60], [128,188], [255, 60]]))
            cv_image_trans = cv2.warpAffine(cv_image_resize, M, (400,300))
            st.write("Image shape (after translation): ", cv_image_trans.shape)
            st.image(cv_image_trans, channels="BGR")

        if st.button("Rotate"):
            st.session_state.state_4 = 4
        if st.session_state.state_4 >= 4:
            # show rotate
            st.subheader("Rotate")
            M = cv2.getRotationMatrix2D(
                    (int(188),int(128)), 10, 0.5)
            cv_image_rotate = cv2.warpAffine(cv_image_trans, M, (400,300))
            st.image(cv_image_rotate, channels="BGR")

        if st.button("Shearing:"):
            st.session_state.state_4 = 5
        if st.session_state.state_4 >= 5:
            # show Shearing
            st.subheader("Shearing")
            M = cv2.getAffineTransform(
                    np.float32([[50,50],  [200,50], [50, 200]]), 
                    np.float32([[10,100], [200,50], [100,250]]))
            cv_image_shearing = cv2.warpAffine(cv_image_rotate, M, (400,300))
            st.image(cv_image_shearing, channels="BGR")
    # ------- end -------

# ------- end -------
# ------- end -------

# ------- HW1 - 5 -------
if topic == '(5) Training Cifar-10 Classifier Using VGG16':
    st.header("Training Cifar-10 Classifier Using VGG16")

    # ------- Upload a Picture -------
    image_0 = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if image_0 is not None:
        file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
        cv_image_0 = cv2.imdecode(file_bytes, 1)

        if st.button("Show Image Detail:"):
            st.write("  Shape: ", cv_image_0.shape)
            st.write("  Size: ", cv_image_0.size)
            st.write("  Data Type: ", cv_image_0.dtype)
            st.write("  Height: ", cv_image_0.shape[0])
            st.write("  Width: ", cv_image_0.shape[1])
        # show image
        st.image(cv_image_0, channels="BGR")
        st.write( torch.cuda.is_available() )
        st.write( torch.cuda.device[0] )
        
    # ------- end -------

# ------- end -------
# ------- end -------

