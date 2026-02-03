import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BISINDO Gesture Detection",
    page_icon="🤟",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center;'>🤟 BISINDO Gesture Detection</h1>
    <p style='text-align: center; font-size:18px;'>
    Deteksi Bahasa Isyarat Indonesia menggunakan <b>YOLOv11</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan Model")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

iou_thres = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

source_option = st.sidebar.radio(
    "📥 Pilih Input",
    ["Upload Gambar", "Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model:** YOLOv11l  
    **Task:** Object Detection  
    **Dataset:** BISINDO v16  
    """
)

# =========================
# IMAGE UPLOAD
# =========================
if source_option == "Upload Gambar":
    st.subheader("📸 Upload Gambar")

    uploaded_file = st.file_uploader(
        "Upload gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Gambar Input", use_column_width=True)

        with col2:
            with st.spinner("🔍 Mendeteksi gesture..."):
                results = model.predict(
                    source=img_array,
                    conf=conf_thres,
                    iou=iou_thres,
                    verbose=False
                )

                annotated = results[0].plot()
                st.image(annotated, caption="Hasil Deteksi", use_column_width=True)

            # Show detected classes
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"✅ Terdeteksi {len(boxes)} gesture")
                for cls in boxes.cls:
                    st.write(f"- {model.names[int(cls)]}")
            else:
                st.warning("⚠️ Tidak ada gesture terdeteksi")

# =========================
# WEBCAM
# =========================
elif source_option == "Webcam":
    st.subheader("🎥 Real-time Webcam Detection")

    run = st.checkbox("▶️ Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Webcam tidak dapat diakses")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = model.predict(
                    source=frame_rgb,
                    conf=conf_thres,
                    iou=iou_thres,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                FRAME_WINDOW.image(annotated_frame)

            cap.release()

# =========================
# FOOTER
# =========================
st.divider()
st.markdown(
    """
    <p style='text-align:center; font-size:14px;'>
    🚀 Developed for BISINDO Gesture Recognition<br>
    YOLOv11 · Streamlit · Computer Vision
    </p>
    """,
    unsafe_allow_html=True
)
