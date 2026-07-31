# ======================================================
# Brain Tumor Detection using YOLOv11
# Streamlit Frontend
# ======================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ========================= */
/* MAIN BACKGROUND */
/* ========================= */

.stApp{
    background:#0B0F14;
}

/* ========================= */
/* ALL TEXT */
/* ========================= */

html,
body,
p,
span,
label,
small,
div,
li{
    color:white !important;
    font-family:'IBM Plex Mono', monospace;
}

/* ========================= */
/* HEADINGS */
/* ========================= */

h1,h2,h3,h4,h5,h6{
    color:white !important;
    font-family:'Space Grotesk', sans-serif;
}

/* ========================= */
/* SIDEBAR */
/* ========================= */

section[data-testid="stSidebar"]{
    background:#151B22;
}

/* Sidebar Heading */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6{
    color:white !important;
}

/* Sidebar Text */

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div{
    color:white !important;
}

/* ========================= */
/* FILE UPLOADER */
/* ========================= */

[data-testid="stFileUploader"]{
    color:white !important;
}

[data-testid="stFileUploader"] *{
    color:white !important;
}

[data-testid="stFileUploaderDropzone"]{
    background:#151B22;
    border:2px dashed #00D4FF;
    border-radius:12px;
}

[data-testid="stFileUploaderDropzone"] *{
    color:white !important;
}

/* ========================= */
/* SLIDER */
/* ========================= */

.stSlider label{
    color:white !important;
}

/* ========================= */
/* BUTTON */
/* ========================= */

.stButton>button{
    background:#00D4FF;
    color:black !important;
    border:none;
    border-radius:10px;
    height:50px;
    width:100%;
    font-weight:bold;
    font-size:16px;
}

.stButton>button:hover{
    background:#45E7FF;
    color:black !important;
}

/* ========================= */
/* METRIC */
/* ========================= */

div[data-testid="stMetric"]{
    background:#151B22;
    padding:15px;
    border-radius:12px;
    border:1px solid #00D4FF;
}

/* ========================= */
/* DATAFRAME */
/* ========================= */

[data-testid="stDataFrame"]{
    background:#151B22;
    border-radius:10px;
}

/* ========================= */
/* ALERTS */
/* ========================= */

.stAlert{
    color:white !important;
}

/* ========================= */
/* CAPTION */
/* ========================= */

.stCaption{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)
# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ======================================================
# HEADER
# ======================================================

st.title("🧠 Brain Tumor Detection using YOLOv11")

st.write("""
Upload a **Brain MRI Image** and let the AI detect brain tumors.

### Supported Formats

- JPG
- JPEG
- PNG

The model has been trained using **YOLOv11 Object Detection**.
""")

st.divider()

# ======================================================
# LAYOUT
# ======================================================

left, right = st.columns(2)

# ======================================================
# LEFT PANEL
# ======================================================

with left:

    st.subheader("Upload MRI Scan")

    uploaded_image = st.file_uploader(
        "Choose MRI Image",
        type=["jpg","jpeg","png"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        0.90,
        0.25,
        0.05
    )

    detect = st.button(
        "🔍 Detect Brain Tumor",
        use_container_width=True
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )

# ======================================================
# RIGHT PANEL
# ======================================================

with right:

    st.subheader("Detection Result")

    if detect:

        if uploaded_image is None:

            st.warning("⚠ Please upload an MRI image.")

        else:

            with st.spinner("Analyzing MRI Scan..."):

                image_np = np.array(image)

                results = model.predict(
                    source=image_np,
                    conf=confidence,
                    verbose=False
                )

                annotated = results[0].plot()

                boxes = results[0].boxes

                st.image(
                    annotated,
                    caption="Detected Tumor",
                    use_container_width=True
                )

                if len(boxes) == 0:

                    st.success("✅ No Brain Tumor Detected")

                else:

                    names = model.names

                    detections = []

                    for box in boxes:

                        cls = int(box.cls[0])

                        conf = float(box.conf[0])

                        detections.append({
                            "Tumor Type": names[cls],
                            "Confidence": f"{conf*100:.2f}%"
                        })

                    df = pd.DataFrame(detections)

                    st.error(f"⚠ {len(df)} Tumor(s) Detected")

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    st.metric(
                        label="Total Tumors",
                        value=len(df)
                    )

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🧠 Model Information")

st.sidebar.write("""
### Model

YOLOv11

### Task

Brain Tumor Detection

### Classes

- Astrocytoma
- Glioblastoma

### Framework

- Streamlit
- Ultralytics YOLO
- Python
""")

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption("""
⚠ **Disclaimer**

This AI model is developed for educational and research purposes only.

It should NOT be used as a replacement for professional medical diagnosis.

Always consult a qualified neurologist or radiologist before making any medical decision.
""")