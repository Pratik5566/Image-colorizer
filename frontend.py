import streamlit as st
import numpy as np
import cv2
from PIL import Image

def colorize_image(image_path):
    prototxt_path = 'colorization_deploy_v2.prototxt'
    model_path = 'colorization_release_v2.caffemodel'
    kernel_path = 'pts_in_hull.npy'

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    colorized = (255.0 * colorized).astype("uint8")

    return bw_image, colorized

def main():
    st.title("Image Colorization App")

    uploaded_file = st.file_uploader("Choose a black and white image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save the image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        bw_image, colorized = colorize_image(temp_image_path)

        st.subheader("Colorized Image")
        st.image(colorized, use_column_width=True)

if __name__ == "__main__":
    main()
