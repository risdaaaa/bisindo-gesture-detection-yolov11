import streamlit as st
import numpy as np
from PIL import Image
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
    return YOLO("best.pt")  # pastikan file ini ada di repo

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🤟 BISINDO Gesture Detection</h1>
    <p style="text-align:center; font-size:18px;">
    Deteksi Bahasa Isyarat Indonesia menggunakan <b>YOLOv11</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model**: YOLOv11l  
    **Task**: Object Detection  
    **Dataset**: BISINDO v16  
    **Deployment**: Streamlit Cloud (CPU)
    """
)

# =====================================================
# MAIN CONTENT
# =====================================================
st.subheader("📸 Upload Gambar")

uploaded_file = st.file_uploader(
    "Upload gambar gesture BISINDO (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="🖼️ Gambar Input",
            use_column_width=True
        )

    with col2:
        with st.spinner("🔍 Mendeteksi gesture..."):
            start_time = time.time()

            results = model.predict(
                source=img_array,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )

            inference_time = time.time() - start_time
            annotated = results[0].plot()

            st.image(
                annotated,
                caption=f"✅ Hasil Deteksi (Inference {inference_time:.2f}s)",
                use_column_width=True
            )

        # =============================
        # DETECTION SUMMARY
        # =============================
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            st.success(f"🎯 Terdeteksi {len(boxes)} gesture")

            detected_classes = []
            for cls_id in boxes.cls:
                detected_classes.append(model.names[int(cls_id)])

            st.markdown("### 🧠 Gesture Terdeteksi:")
            for name in sorted(set(detected_classes)):
                st.markdown(f"- **{name}**")
        else:
            st.warning("⚠️ Tidak ada gesture terdeteksi")

else:
    st.info("⬆️ Silakan upload gambar untuk memulai deteksi.")

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown(
    """
    <p style="text-align:center; font-size:14px;">
    🚀 BISINDO Gesture Detection System<br>
    YOLOv11 · PyTorch · Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
