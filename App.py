import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image
from io import BytesIO
import base64

# ---------------------- Haar Cascade ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------- Face Detection for Images ----------------------
def detect_faces_image(image, scale_factor=1.1, min_neighbors=3):
    """
    Detect faces in an image with adjustable parameters.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (95, 207, 30), 3)
        cv2.rectangle(image, (x, y - 40), (x + w, y), (95, 207, 30), -1)
        cv2.putText(image, f'F-{i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return image, len(faces)

# ---------------------- Face Detection for Webcam ----------------------
class VideoFaceTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,   # Optimized for webcam
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for i, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(img, (x, y), (x+w, y+h), (95, 207, 30), 3)
            cv2.rectangle(img, (x, y-40), (x+w, y), (95, 207, 30), -1)
            cv2.putText(img, f'F-{i}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return img

# ---------------------- Helper: Download Image ----------------------
def get_download_link(img, filename="output.jpg"):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download Processed Image</a>'

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("Face Detection using OpenCV & Streamlit")

# Sidebar
mode = st.sidebar.selectbox("Select Input Source", ["Image", "Webcam"])
st.sidebar.markdown("[© Developed by Prakhar Barsainya](https://github.com/PrakharBarsainya)")

# ---------------------- IMAGE MODE ----------------------
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(img, caption="Original Image", use_column_width=True)

        st.markdown("### Adjust Face Detection Parameters")
        scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
        min_neighbors = st.slider("Min Neighbors", 1, 6, 3, 1)

        detected_img, count = detect_faces_image(img.copy(), scale_factor, min_neighbors)
        st.image(detected_img, caption=f"Detected Faces: {count}", use_column_width=True)

        if count == 0:
            st.error("No faces detected!")
        else:
            st.success(f"Total Faces Detected: {count}")
            st.markdown(get_download_link(detected_img, uploaded_file.name), unsafe_allow_html=True)

# ---------------------- WEBCAM MODE ----------------------
elif mode == "Webcam":
    st.markdown("### Webcam Face Detection (desktop preferred)")
    webrtc_streamer(
        key="webcam",
        video_transformer_factory=VideoFaceTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=False  # <- ensures stable webcam frames
    )
