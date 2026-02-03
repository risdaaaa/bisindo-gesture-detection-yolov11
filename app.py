import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="BISINDO Gesture Detection",
    page_icon="🤟",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # pastikan file model ada di repo

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🤟 BISINDO Gesture Detection</h1>
    <p style="text-align:center; font-size:18px;">
    Deteksi Bahasa Isyarat Indonesia menggunakan <b>YOLOv11</b><br>
    Input: <b>Foto (Image Upload)</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan Deteksi")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.4, 0.05
)

iou_thres = st.sidebar.slider(
    "IoU Threshold",
    0.1, 1.0, 0.5, 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model**: YOLOv11l  
    **Dataset**: BISINDO v16  
    **Mode**: Image Upload Only  
    **Platform**: Streamlit Cloud
    """
)

# =====================================================
# MAIN CONTENT
# =====================================================
st.subheader("📸 Upload Foto Gesture BISINDO")

uploaded_file = st.file_uploader(
    "Upload gambar (.jpg / .png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    # =====================
    # ORIGINAL IMAGE
    # =====================
    with col1:
        st.image(
            image,
            caption="🖼️ Gambar Asli",
            use_column_width=True
        )

    # =====================
    # DETECTION
    # =====================
    with col2:
        with st.spinner("🔍 Mendeteksi gesture..."):
            start_time = time.time()

            results = model.predict(
                source=img_np,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )

            inference_time = time.time() - start_time

            draw = ImageDraw.Draw(image)
            boxes = results[0].boxes
            detected_labels = []

            if boxes is not None:
                for box, cls, conf in zip(
                    boxes.xyxy,
                    boxes.cls,
                    boxes.conf
                ):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label = model.names[int(cls)]
                    detected_labels.append(label)

                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline="lime",
                        width=3
                    )
                    draw.text(
                        (x1, y1 - 12),
                        f"{label} {conf:.2f}",
                        fill="lime"
                    )

            st.image(
                image,
                caption=f"✅ Hasil Deteksi (Inference {inference_time:.2f} detik)",
                use_column_width=True
            )

            if detected_labels:
                st.success(f"🎯 Terdeteksi {len(detected_labels)} gesture")
                st.markdown("### 🧠 Gesture Terdeteksi:")
                for lbl in sorted(set(detected_labels)):
                    st.markdown(f"- **{lbl}**")
            else:
                st.warning("⚠️ Tidak ada gesture terdeteksi")

else:
    st.info("⬆️ Silakan upload foto gesture BISINDO untuk memulai deteksi.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown(
    """
    <p style="text-align:center; font-size:14px;">
    YOLOv11 · BISINDO Gesture Detection<br>
    Streamlit Cloud Deployment
    </p>
    """,
    unsafe_allow_html=True
)
