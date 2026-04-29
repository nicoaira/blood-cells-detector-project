"""
🔬 Blood Cell Detector — Streamlit Web App
============================================
Detect and classify blood cells in peripheral blood smear images
using a fine-tuned YOLO26 model.

Classes: RBC 🔴 | Platelets 🟢 | Neutrophil 🔵 | Lymphocyte 💜
         Monocyte 🟤 | Eosinophil 🟠 | Basophil 🟣
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
METADATA_PATH = ROOT / "blood_detector_metadata.json"
TEST_IMAGES_DIR = ROOT / "test_images"

# ─── Constants ────────────────────────────────────────────────────────────────
WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# BGR colors for OpenCV drawing
CLASS_COLORS_BGR = {
    "RBC":        (60,  60,  220),   # deep red
    "Platelets":  (50,  200, 80),    # green
    "Neutrophil": (230, 140, 40),    # blue
    "Lymphocyte": (200, 80,  180),   # purple
    "Monocyte":   (80,  140, 200),   # brown/amber
    "Eosinophil": (50,  160, 240),   # orange
    "Basophil":   (180, 60,  200),   # magenta
}

# Emoji for each class
CLASS_EMOJI = {
    "RBC":        "🔴",
    "Platelets":  "🟢",
    "Neutrophil": "🔵",
    "Lymphocyte": "💜",
    "Monocyte":   "🟤",
    "Eosinophil": "🟠",
    "Basophil":   "🟣",
}

# RGB colors for metric cards (CSS)
CLASS_COLORS_HEX = {
    "RBC":        "#DC3C3C",
    "Platelets":  "#50C850",
    "Neutrophil": "#288CE6",
    "Lymphocyte": "#B450C8",
    "Monocyte":   "#C88C50",
    "Eosinophil": "#F0A032",
    "Basophil":   "#C83CB4",
}

# Display order
CLASS_ORDER = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
               "Monocyte", "Eosinophil", "Basophil"]


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔬 Blood Cell Detector",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS (dark theme enhancements) ────────────────────────────────────
st.markdown("""
<style>
/* ── Global tweaks ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid rgba(88, 166, 255, 0.15);
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(88,166,255,0.12) 0%, rgba(200,80,200,0.10) 100%);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero-header h1 {
    font-size: 2.4rem;
    background: linear-gradient(90deg, #58a6ff, #bc8cff, #f778ba);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-header p {
    color: #8b949e;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Metric card ── */
.metric-card {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.3s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(88,166,255,0.4);
}
.metric-card .emoji { font-size: 1.6rem; }
.metric-card .count {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.2rem 0;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #8b949e;
}

/* ── Stats banner ── */
.stats-banner {
    background: linear-gradient(90deg, rgba(88,166,255,0.08), rgba(200,80,200,0.08));
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-around;
    margin: 1rem 0 1.5rem 0;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.stats-banner .stat {
    text-align: center;
    min-width: 100px;
}
.stats-banner .stat .num {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stats-banner .stat .lbl {
    color: #8b949e;
    font-size: 0.8rem;
}

/* ── Info box ── */
.info-box {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #c9d1d9;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #484f58;
    font-size: 0.78rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(88,166,255,0.08);
    margin-top: 3rem;
}
.footer a { color: #58a6ff; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the YOLO model once and cache it."""
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    return model


# ─── Drawing helpers ──────────────────────────────────────────────────────────
def get_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(name, (200, 200, 200))


def draw_detections(img: np.ndarray, boxes, classes, confs, names,
                    line_width: int = 2, font_scale: float = 0.45) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    ft = 1

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
        name = names[int(cls_id)]
        col = get_color(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, line_width)

        # Label background
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, ft)
        ly = y1 - 4
        if ly - th - 4 < 0:
            ly = y2 + th + 6

        cv2.rectangle(annotated, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), col, -1)
        cv2.putText(annotated, label, (x1 + 3, ly - 1), font, font_scale,
                    (255, 255, 255), ft, cv2.LINE_AA)

    return annotated


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    conf_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Minimum confidence score to display a detection."
    )

    iou_threshold = st.slider(
        "📐 IoU Threshold (NMS)",
        min_value=0.1, max_value=0.95, value=0.7, step=0.05,
        help="Intersection over Union for Non-Maximum Suppression."
    )

    img_size = st.select_slider(
        "📏 Inference Size",
        options=[320, 480, 640, 800, 1024],
        value=640,
        help="Image size for inference. 640 is the training resolution."
    )

    st.markdown("---")
    st.markdown("### 🎨 Display Options")

    line_width = st.slider("Box Thickness", 1, 5, 2)
    font_scale = st.slider("Label Size", 0.3, 1.0, 0.45, 0.05)

    show_class_filter = st.multiselect(
        "🔍 Show Only Classes",
        options=CLASS_ORDER,
        default=CLASS_ORDER,
        help="Filter which classes to display."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    with st.expander("🧠 Architecture Details", expanded=False):
        if METADATA_PATH.exists():
            meta = json.loads(METADATA_PATH.read_text())
            st.markdown(f"""
- **Model:** YOLO26m
- **Task:** Object Detection
- **Input:** 640×640
- **Classes:** {meta.get('nc', 7)}
- **Ultralytics:** v{meta.get('ultralytics_version_trained', 'N/A')}
""")
        else:
            st.info("Metadata file not found.")

    with st.expander("📈 Validation Metrics", expanded=False):
        st.markdown("""
| Metric | Value |
|--------|-------|
| Precision | 0.850 |
| Recall | 0.848 |
| mAP@50 | 0.875 |
| mAP@50–95 | 0.812 |
""")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#484f58;font-size:0.75rem;">'
        '🔬 Built with Ultralytics & Streamlit<br>'
        '🩸 Blood Cell Detector v1.0'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Hero header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔬 Blood Cell Detector 🩸</h1>
    <p>🧬 AI-powered detection and classification of blood cells in peripheral smear images</p>
</div>
""", unsafe_allow_html=True)


# ─── Input section ────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload Your Image")

col_upload, col_or, col_sample = st.columns([5, 1, 5])

with col_upload:
    uploaded_file = st.file_uploader(
        "Drop a blood smear image here 🖼️",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a peripheral blood smear image for analysis.",
        label_visibility="collapsed",
    )

with col_or:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'height:100%;padding-top:1rem;">'
        '<span style="color:#484f58;font-size:1.1rem;font-weight:600;">OR</span></div>',
        unsafe_allow_html=True,
    )

with col_sample:
    sample_images = sorted(TEST_IMAGES_DIR.glob("*.png")) if TEST_IMAGES_DIR.exists() else []
    sample_names = ["— Select a sample —"] + [p.stem.replace("_", " ").title() for p in sample_images]
    selected_sample = st.selectbox(
        "🧪 Try a sample image",
        options=range(len(sample_names)),
        format_func=lambda i: sample_names[i],
        label_visibility="collapsed",
    )


# ─── Determine input image ───────────────────────────────────────────────────
input_image = None
source_label = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    source_label = f"📤 Uploaded: {uploaded_file.name}"
elif selected_sample > 0:
    sample_path = sample_images[selected_sample - 1]
    input_image = Image.open(sample_path).convert("RGB")
    source_label = f"🧪 Sample: {sample_path.stem}"


# ─── Run detection ────────────────────────────────────────────────────────────
if input_image is not None:
    st.markdown("---")

    # Convert to numpy (BGR for OpenCV)
    img_np = np.array(input_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Run inference
    with st.spinner("🔬 Analyzing blood smear... 🧬"):
        model = load_model()
        results = model.predict(
            source=img_np,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            device="cpu",
            verbose=False,
        )

    r = results[0]
    all_boxes = r.boxes.xyxy.cpu().numpy()
    all_classes = r.boxes.cls.cpu().numpy().astype(int)
    all_confs = r.boxes.conf.cpu().numpy()
    names = r.names

    # Filter by selected classes
    mask = np.array([names[c] in show_class_filter for c in all_classes])
    boxes = all_boxes[mask]
    classes = all_classes[mask]
    confs = all_confs[mask]

    # Count
    counts = Counter(names[int(c)] for c in classes)
    total = len(boxes)

    # ─── Stats banner ─────────────────────────────────────────────────────
    total_wbc = sum(counts.get(w, 0) for w in WBC_SUBTYPES)
    st.markdown(f"""
    <div class="stats-banner">
        <div class="stat">
            <div class="num">{total}</div>
            <div class="lbl">🔬 Total Cells</div>
        </div>
        <div class="stat">
            <div class="num">{counts.get("RBC", 0)}</div>
            <div class="lbl">🔴 Red Blood Cells</div>
        </div>
        <div class="stat">
            <div class="num">{total_wbc}</div>
            <div class="lbl">⚪ White Blood Cells</div>
        </div>
        <div class="stat">
            <div class="num">{counts.get("Platelets", 0)}</div>
            <div class="lbl">🟢 Platelets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Image display ────────────────────────────────────────────────────
    st.markdown(f"#### 🖼️ Detection Results — {source_label}")

    annotated_bgr = draw_detections(
        img_bgr, boxes, classes, confs, names,
        line_width=line_width, font_scale=font_scale,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown("**📷 Original Image**")
        st.image(input_image, use_container_width=True)

    with col_det:
        st.markdown("**🔬 Detected Cells**")
        st.image(annotated_rgb, use_container_width=True)

    # ─── Per-class breakdown ──────────────────────────────────────────────
    st.markdown("#### 📊 Cell Count Breakdown")

    cols = st.columns(len(CLASS_ORDER))
    for i, cls_name in enumerate(CLASS_ORDER):
        count = counts.get(cls_name, 0)
        emoji = CLASS_EMOJI[cls_name]
        color = CLASS_COLORS_HEX[cls_name]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="emoji">{emoji}</div>
                <div class="count" style="color:{color};">{count}</div>
                <div class="label">{cls_name}</div>
            </div>
            """, unsafe_allow_html=True)

    # ─── WBC Differential ─────────────────────────────────────────────────
    if total_wbc > 0:
        st.markdown("#### 🧬 WBC Differential")
        st.markdown(
            '<div class="info-box">💡 The WBC differential shows the relative '
            'proportion of each white blood cell subtype. This is a key diagnostic '
            'metric in hematology.</div>',
            unsafe_allow_html=True,
        )

        diff_cols = st.columns(5)
        for i, wbc in enumerate(["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]):
            c = counts.get(wbc, 0)
            pct = (c / total_wbc * 100) if total_wbc > 0 else 0
            emoji = CLASS_EMOJI[wbc]
            color = CLASS_COLORS_HEX[wbc]
            with diff_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="emoji">{emoji}</div>
                    <div class="count" style="color:{color};">{pct:.1f}%</div>
                    <div class="label">{wbc} ({c})</div>
                </div>
                """, unsafe_allow_html=True)

    # ─── Download annotated image ─────────────────────────────────────────
    st.markdown("---")

    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        # Encode annotated image to bytes
        annotated_pil = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="💾 Download Annotated Image",
            data=buf,
            file_name="blood_cell_detection.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    # ─── Welcome state ────────────────────────────────────────────────────
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:2rem;">
            <p style="font-size:3rem; margin:0;">🩸🔬🧬</p>
            <p style="font-size:1.2rem; color:#c9d1d9; margin:0.5rem 0;">
                Upload a blood smear image or select a sample to get started!
            </p>
            <p style="font-size:0.85rem; color:#8b949e; margin:0;">
                The model detects <strong>7 cell types</strong>: RBC 🔴, Platelets 🟢,
                Neutrophil 🔵, Lymphocyte 💜, Monocyte 🟤, Eosinophil 🟠, Basophil 🟣
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("### ✨ Features")
    feat_cols = st.columns(3)

    with feat_cols[0]:
        st.markdown("""
        <div class="metric-card" style="padding:1.5rem;">
            <div class="emoji" style="font-size:2.5rem;">🎯</div>
            <div class="label" style="font-size:1rem; color:#c9d1d9; margin-top:0.8rem;">
                <strong>High Accuracy</strong><br>
                <span style="color:#8b949e;">mAP@50 of 87.5% on validation set</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with feat_cols[1]:
        st.markdown("""
        <div class="metric-card" style="padding:1.5rem;">
            <div class="emoji" style="font-size:2.5rem;">⚡</div>
            <div class="label" style="font-size:1rem; color:#c9d1d9; margin-top:0.8rem;">
                <strong>Fast Inference</strong><br>
                <span style="color:#8b949e;">Powered by YOLO26 architecture</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with feat_cols[2]:
        st.markdown("""
        <div class="metric-card" style="padding:1.5rem;">
            <div class="emoji" style="font-size:2.5rem;">🧬</div>
            <div class="label" style="font-size:1rem; color:#c9d1d9; margin-top:0.8rem;">
                <strong>7 Cell Types</strong><br>
                <span style="color:#8b949e;">Full differential WBC count</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔬 Blood Cell Detector v1.0 · Built with
    <a href="https://ultralytics.com">Ultralytics</a> &
    <a href="https://streamlit.io">Streamlit</a> ·
    ⚠️ For research purposes only — not for clinical use
</div>
""", unsafe_allow_html=True)
