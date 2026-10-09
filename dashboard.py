"""SAFE VISION AI - Intelligent Road Safety & Traffic Monitoring System.

Run with:
    streamlit run dashboard.py

The dashboard works in demo mode when optional model/media assets are missing.
Uploaded images and videos use the available YOLO weights for real inference.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

try:
    import av
    from streamlit_webrtc import webrtc_streamer
except ImportError:
    av = None
    webrtc_streamer = None


ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Traffic Detection": ROOT / "runs" / "traffic_detection_finetune_10epochs" / "weights" / "best.pt",
    "Helmet Detection": ROOT / "runs" / "helmet_detection_10epochs" / "weights" / "best.pt",
    "Emergency Vehicle Detection": ROOT / "runs" / "emergency_vehicle_detection_10epochs" / "weights" / "best.pt",
}

NAV_ITEMS = {
    "Command Center": ("⌂", "Overview"),
    "Video Detection": ("▣", "Analyze media"),
    "Live Camera": ("◉", "Camera stream"),
    "Traffic Violations": ("△", "Safety intelligence"),
    "Emergency Vehicle Detection": ("!", "Priority response"),
    "Number Plate Recognition": ("▤", "Plate intelligence"),
    "Analytics": ("◌", "Performance insights"),
    "Vehicle Tracking": ("⌖", "Live fleet"),
    "Incident Reports": ("≡", "Incident center"),
    "System Settings": ("⚙", "Configuration"),
}

DEMO_STATS = {
    "Total Vehicles": ("2,847", "12.8%", "Vehicles tracked today", "blue"),
    "Motorcycles": ("682", "8.4%", "Two-wheelers detected", "violet"),
    "Cars": ("1,904", "5.2%", "Passenger vehicles", "cyan"),
    "Heavy Vehicles": ("261", "3.1%", "Buses and trucks", "amber"),
    "Helmet Violations": ("43", "−14.6%", "Requires review", "red"),
    "Triple Riding": ("17", "−8.2%", "Safety violation", "orange"),
    "Wrong-Way": ("09", "−2.4%", "Active alerts", "red"),
    "Emergency Vehicles": ("06", "33.3%", "Priority events", "green"),
}


st.set_page_config(
    page_title="SAFE VISION AI | Traffic Command Center",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --bg:#071019; --panel:rgba(15,29,42,.76); --line:rgba(150,190,215,.14); --muted:#8193a4; --text:#edf6fb; --cyan:#43d9df; --blue:#5a8cff; --red:#ff647a; --amber:#f5b74c; }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp { background: radial-gradient(circle at 72% -10%, #12314b 0, transparent 34%), radial-gradient(circle at 4% 44%, #102b37 0, transparent 26%), var(--bg); color:var(--text); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#08131f 0%,#091723 100%); border-right:1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding:1.4rem 1rem; }
        .block-container { max-width:1560px; padding:1.4rem 2.5rem 3rem; }
        .brand-mark { display:flex; align-items:center; gap:11px; margin-bottom:2.25rem; }
        .brand-orb { width:35px; height:35px; border:1px solid #47dfe1; border-radius:11px; display:grid; place-items:center; color:#52edf0; box-shadow:0 0 22px #32dae944; font-weight:800; }
        .brand-name { font-size:1rem; letter-spacing:.15em; font-weight:800; color:#f4fbff; }
        .brand-sub { color:#6f8495; font-size:.61rem; letter-spacing:.08em; margin-top:2px; }
        .nav-label { color:#5f7486; text-transform:uppercase; letter-spacing:.15em; font-size:.62rem; margin:0 0 .65rem .4rem; }
        .nav-item { padding:.72rem .75rem; margin:.18rem 0; border-radius:10px; color:#91a5b5; font-size:.79rem; display:flex; gap:.75rem; align-items:center; }
        .nav-item.active { color:#ecffff; background:linear-gradient(90deg,#17435699,#13304033); border:1px solid #41d5db38; box-shadow:inset 3px 0 #43d9df; }
        .nav-icon { color:#5d8095; font-family:'DM Mono'; width:16px; text-align:center; } .active .nav-icon { color:#55edf0; }
        .side-foot { margin-top:2rem; padding:.85rem; border:1px solid var(--line); border-radius:12px; background:#0c1b28; }
        .eyebrow { color:#52dfe4; font-size:.66rem; font-family:'DM Mono'; letter-spacing:.14em; text-transform:uppercase; }
        .hero-title { font-size:2rem; line-height:1.1; margin:.32rem 0 .5rem; font-weight:800; letter-spacing:-.04em; }
        .hero-copy { color:#8295a5; font-size:.82rem; max-width:660px; line-height:1.6; }
        .header-shell { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; margin-bottom:1.5rem; }
        .live-pills { display:flex; gap:.55rem; flex-wrap:wrap; justify-content:flex-end; }
        .pill { padding:.5rem .72rem; border:1px solid var(--line); border-radius:9px; color:#a8bccb; background:#0c1c29b3; font-size:.68rem; white-space:nowrap; }
        .dot { display:inline-block; width:6px; height:6px; background:#53e0ba; border-radius:50%; margin-right:6px; box-shadow:0 0 9px #53e0ba; }
        .panel { border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.25rem; background:linear-gradient(145deg,rgba(18,37,53,.79),rgba(8,20,31,.74)); box-shadow:0 16px 45px #00000018; }
        .panel-title { font-size:.79rem; text-transform:uppercase; letter-spacing:.11em; font-weight:700; color:#c1d4df; margin-bottom:.2rem; }
        .panel-sub { color:#718797; font-size:.68rem; margin-bottom:1rem; }
        .metric { border:1px solid var(--line); border-radius:13px; padding:.9rem 1rem; background:linear-gradient(145deg,#122638c9,#0b1926b8); min-height:112px; position:relative; overflow:hidden; }
        .metric:after { content:''; position:absolute; width:75px; height:75px; right:-30px; top:-35px; border-radius:50%; background:#43d9df0d; }
        .metric-label { color:#8da2b0; font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; }
        .metric-value { font-size:1.65rem; font-weight:800; margin:.25rem 0 .1rem; letter-spacing:-.04em; }
        .metric-meta { color:#6e8493; font-size:.67rem; } .metric-change { color:#59dbbc; font-family:'DM Mono'; font-size:.67rem; }
        .metric-icon { float:right; color:#52dfe4; font-size:1.1rem; }
        .section { margin-top:1.35rem; margin-bottom:.7rem; display:flex; align-items:end; justify-content:space-between; }
        .section h2 { font-size:1rem; margin:0; letter-spacing:-.02em; } .section p { color:#718797; font-size:.7rem; margin:.25rem 0 0; }
        .monitor { min-height:390px; background:linear-gradient(135deg,#071923,#122c38 48%,#0b1a27); position:relative; overflow:hidden; }
        .monitor-grid { position:absolute; inset:0; opacity:.18; background-image:linear-gradient(#67dfe326 1px,transparent 1px),linear-gradient(90deg,#67dfe326 1px,transparent 1px); background-size:42px 42px; }
        .monitor-scan { position:absolute; left:0; right:0; top:40%; height:1px; background:#51e6e755; box-shadow:0 0 18px #51e6e7; animation:scan 5s linear infinite; }
        @keyframes scan { 0%{top:18%} 50%{top:78%} 100%{top:18%} }
        .monitor-content { position:relative; z-index:1; height:100%; min-height:365px; display:flex; flex-direction:column; justify-content:space-between; }
        .monitor-top { display:flex; justify-content:space-between; align-items:center; } .ai-badge { color:#6af0d0; font-family:'DM Mono'; font-size:.66rem; letter-spacing:.08em; }
        .ai-badge:before { content:' '; display:inline-block; width:7px; height:7px; border-radius:50%; background:#62e6c7; margin-right:7px; box-shadow:0 0 12px #62e6c7; }
        .camera-empty { margin:auto; text-align:center; color:#7f9aa8; } .camera-glyph { color:#3ed0d9; font-size:3.3rem; opacity:.75; }
        .camera-empty strong { display:block; color:#d5e8ed; font-size:.9rem; margin-top:.5rem; } .camera-empty span { display:block; font-size:.7rem; margin-top:.3rem; }
        .telemetry { display:flex; gap:1.25rem; font-family:'DM Mono'; color:#91aebd; font-size:.67rem; } .telemetry b { color:#e1f6f8; font-weight:500; }
        .status-row { display:flex; align-items:center; gap:.4rem; font-size:.7rem; } .status-ok { color:#59dcbd; } .status-warn { color:#f3c269; } .status-bad { color:#ff7183; }
        .tag { border-radius:5px; padding:.22rem .42rem; font-size:.61rem; font-family:'DM Mono'; } .tag-high { color:#ff8391; background:#ff637a18; border:1px solid #ff637a35; } .tag-medium { color:#f2c36e; background:#f4b94e14; border:1px solid #f4b94e33; } .tag-low { color:#61dcc1; background:#5de0bd14; border:1px solid #5de0bd33; }
        .mini-stat { display:flex; justify-content:space-between; padding:.72rem 0; border-bottom:1px solid #ffffff0a; font-size:.73rem; color:#8ba0ae; } .mini-stat:last-child { border:0; } .mini-stat strong { color:#ecf7fa; font-weight:600; }
        .health-card { display:flex; gap:.6rem; align-items:center; padding:.7rem .75rem; border:1px solid var(--line); border-radius:10px; background:#0b1d2a; font-size:.7rem; color:#9eb2bf; } .health-card b { display:block; color:#e8f7fa; font-size:.73rem; margin-bottom:.16rem; } .health-dot { width:7px; height:7px; background:#54ddbd; border-radius:50%; box-shadow:0 0 9px #54ddbd; }
        .demo-note { color:#d3ae6a; font-size:.68rem; padding:.55rem .75rem; border:1px solid #c9973b33; background:#c9973b0d; border-radius:8px; }
        .footer { color:#5f7786; font-size:.67rem; border-top:1px solid var(--line); margin-top:2rem; padding-top:1rem; display:flex; justify-content:space-between; }
        div[data-testid="stMetric"] { background:transparent; }
        .stButton > button { border-radius:9px; border:1px solid #42d8df66; background:#123746; color:#dffeff; font-weight:600; }
        .stButton > button:hover { border-color:#6becef; background:#174a59; }
        [data-testid="stSidebar"] .stButton > button { border:1px solid transparent; background:transparent; color:#91a5b5; text-align:left; justify-content:flex-start; min-height:38px; padding:.35rem .7rem; font-size:.76rem; font-weight:500; }
        [data-testid="stSidebar"] .stButton > button:hover { border-color:#2c6575; background:#123342; color:#e7ffff; }
        [data-testid="stFileUploader"] { border:1px dashed #4bcbd455; border-radius:11px; background:#0c2130; } .stTabs [data-baseweb="tab-list"] { gap:1.2rem; } .stTabs [aria-selected="true"] { color:#53e0e5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def panel_heading(title: str, subtitle: str = "") -> None:
    html(f'<div class="panel-title">{title}</div><div class="panel-sub">{subtitle}</div>')


def status_dot(status: str = "ok") -> str:
    return f'<span class="dot" style="background:var(--{"red" if status == "bad" else "amber" if status == "warn" else "cyan"});box-shadow:0 0 9px var(--{"red" if status == "bad" else "amber" if status == "warn" else "cyan"});"></span>'


def metric_card(label: str, value: str, meta: str, change: str, icon: str, tone: str) -> None:
    state = "LIVE" if change == "LIVE" else "DEMO"
    state_class = "status-ok" if state == "LIVE" else "status-warn"
    html(
        f'<div class="metric"><span class="metric-icon">{icon}</span>'
        f'<div class="metric-label">{label} <span class="status-row {state_class}" style="float:right;font-size:.58rem">{status_dot("ok" if state == "LIVE" else "warn")}{state}</span></div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-meta">{meta}</div><div class="metric-change">{change if state == "LIVE" else change} <span style="color:#627b89">vs. previous period</span></div></div>'
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Any:
    if YOLO is None:
        raise RuntimeError("The ultralytics package is not installed.")
    return YOLO(model_path)


def model_for(name: str) -> tuple[Any | None, str | None]:
    path = MODEL_PATHS[name]
    if not path.exists():
        return None, f"Model weights are not available at {path}."
    try:
        return load_model(str(path)), None
    except Exception as exc:
        return None, f"Unable to load {name}: {exc}"


def class_name(model: Any, box: Any) -> str:
    class_id = int(box.cls[0].item())
    return str(model.names[class_id])


def predict(model: Any, image: np.ndarray, confidence: float) -> Any:
    results = model.predict(source=image, imgsz=640, conf=confidence, verbose=False)
    return results[0]


def update_detection_state(model: Any, result: Any) -> None:
    counts = Counter(class_name(model, box).lower() for box in result.boxes)
    st.session_state["last_detection_counts"] = counts
    st.session_state["last_detection_objects"] = len(result.boxes)
    st.session_state["last_detection_confidence"] = (
        float(np.mean([float(box.conf[0].item()) for box in result.boxes]))
        if len(result.boxes)
        else 0.0
    )
    st.session_state["last_detection_time"] = datetime.now().strftime("%H:%M:%S")


def render_header() -> None:
    now = datetime.now().strftime("%d %b %Y  •  %H:%M:%S")
    html(
        f'<div class="header-shell"><div><div class="eyebrow">AI TRAFFIC COMMAND CENTER / SECTOR 07</div>'
        f'<div class="hero-title">SAFE VISION AI</div>'
        f'<div class="hero-copy">See Risk. Detect Threats. Protect Lives.<br>'
        f'An intelligent road safety platform for real-time traffic monitoring, violation analysis and priority response.</div></div>'
        f'<div class="live-pills"><div class="pill">{status_dot()}AI SYSTEM <b style="color:#61ddbd">ONLINE</b></div>'
        f'<div class="pill">{status_dot()}CAMERAS <b style="color:#61ddbd">ACTIVE</b></div>'
        f'<div class="pill">ENGINE <b style="color:#d9edf2">YOLO</b></div><div class="pill">UPDATED <b>{now}</b></div></div></div>'
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        '<div class="brand-mark"><div class="brand-orb">SV</div><div><div class="brand-name">SAFE VISION</div><div class="brand-sub">INTELLIGENT ROAD SAFETY</div></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="nav-label">Operations</div>', unsafe_allow_html=True)
    labels = list(NAV_ITEMS)
    current = st.session_state.get("active_page", labels[0])
    for label, (icon, _) in NAV_ITEMS.items():
        prefix = "▸" if label == current else " "
        if st.sidebar.button(f"{prefix} {icon}   {label}", key=f"nav_{label}"):
            current = label
            st.session_state["active_page"] = label
            st.rerun()
    st.sidebar.markdown(
        '<div class="side-foot"><div class="eyebrow">SYSTEM MODE</div><div style="font-size:.78rem;color:#deedf0;margin:.35rem 0">● Monitoring active</div><div style="font-size:.65rem;color:#708796;line-height:1.45">Local inference enabled. External alerts are not connected.</div></div>',
        unsafe_allow_html=True,
    )
    return current


def render_live_camera(model: Any | None, confidence: float) -> None:
    """Real-time webcam detection using streamlit-webrtc."""

    st.markdown(
        '<div class="section">'
        '<div><h2>Live Camera Detection</h2>'
        '<p>Real-time road-safety detection from your webcam.</p></div>'
        '<span class="eyebrow">WEBCAM / YOLO</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if webrtc_streamer is None or av is None:
        st.error("Webcam support is not installed.")
        st.code(
            "python -m pip install -U streamlit-webrtc av",
            language="powershell",
        )
        return

    if cv2 is None:
        st.error("OpenCV is not installed.")
        st.code(
            "python -m pip install -U opencv-python",
            language="powershell",
        )
        return

    if model is None:
        st.error("YOLO model could not be loaded.")
        st.info(
            "Go to System Settings and select a model with a valid best.pt file."
        )
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            padding:12px;
            border-radius:10px;
            background:#0b1d2a;
            border:1px solid rgba(150,190,215,.14);
            margin-bottom:12px;
        ">
            <b style="color:#dffeff;">Camera Instructions</b>
            <div style="
                color:#8da2b0;
                font-size:.72rem;
                margin-top:6px;
                line-height:1.6;
            ">
                1. Click START below.<br>
                2. Allow camera permission in your browser.<br>
                3. Show a road or vehicle scene to the camera.<br>
                4. YOLO will draw detection boxes on the live video.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_model = model
    active_confidence = confidence

    def video_frame_callback(frame: Any) -> Any:
        try:
            image = frame.to_ndarray(format="bgr24")

            results = active_model.predict(
                source=image,
                imgsz=416,
                conf=active_confidence,
                verbose=False,
                device="cpu",
            )

            annotated = results[0].plot()

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )

        except Exception:
            return frame

    webrtc_streamer(
        key="safe-vision-live-camera-v3",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "AI predictions are visual assistance only. Human verification is required."
    )


def render_monitor(model: Any | None, confidence: float, mode: str = "Command Center") -> None:
    """Command-center monitor for uploaded image results."""

    st.markdown('<div class="panel monitor">', unsafe_allow_html=True)
    html(
        '<div class="monitor-grid"></div><div class="monitor-scan"></div>'
        '<div class="monitor-content">'
        '<div class="monitor-top">'
        '<span class="ai-badge">AI DETECTION ACTIVE</span>'
        '<span class="tag tag-low">LOCAL INFERENCE</span>'
        '</div>'
    )

    if "last_annotated" in st.session_state:
        st.image(
            st.session_state["last_annotated"],
            use_container_width=True,
        )
    else:
        html(
            '<div class="camera-empty">'
            '<div class="camera-glyph">◉</div>'
            '<strong>Awaiting an input source</strong>'
            '<span>Upload a road image or video to activate the detection overlay.</span>'
            '</div>'
        )

    objects = st.session_state.get("last_detection_objects", "—")
    conf = st.session_state.get("last_detection_confidence", 0)
    confidence_text = f"{conf:.1%}" if conf else "—"

    html(
        f'<div class="telemetry">'
        f'<span>CONFIDENCE <b>{confidence_text}</b></span>'
        f'<span>OBJECTS <b>{objects}</b></span>'
        f'<span>FRAME <b>{st.session_state.get("last_detection_time", "STANDBY")}</b></span>'
        f'<span style="color:#5de0bd">STATUS <b>READY</b></span>'
        f'</div></div>'
    )

    st.markdown("</div>", unsafe_allow_html=True)


def detection_input(model: Any | None, confidence: float) -> None:
    uploaded = st.file_uploader("Drop an image for live YOLO inference", type=["jpg", "jpeg", "png", "webp"], key="command_image")
    if uploaded is None:
        return
    try:
        image = Image.open(uploaded).convert("RGB")
        if st.button("Run detection", key="command_detect"):
            if model is None:
                st.warning("No compatible model is available. Add the selected best.pt weights to activate real inference.")
                return
            if cv2 is None:
                st.error("OpenCV is required for image inference.")
                return
            result = predict(model, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), confidence)
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            update_detection_state(model, result)
            st.session_state["last_annotated"] = annotated
            st.rerun()
    except Exception as exc:
        st.error(f"Could not process the image: {exc}")


def render_metrics() -> None:
    counts = st.session_state.get("last_detection_counts")
    if counts:
        total = st.session_state.get("last_detection_objects", 0)
        values = {
            "Total Vehicles": (str(total), "From latest inference", "LIVE", "blue"),
            "Motorcycles": (str(counts.get("motorcycle", 0)), "Latest inference", "LIVE", "violet"),
            "Cars": (str(counts.get("car", 0)), "Latest inference", "LIVE", "cyan"),
            "Heavy Vehicles": (str(counts.get("truck", 0) + counts.get("bus", 0)), "Bus + truck classes", "LIVE", "amber"),
            "Helmet Violations": (str(counts.get("no helmet", 0)), "Model class signal", "LIVE", "red"),
            "Triple Riding": ("—", "Requires temporal analysis", "PENDING", "orange"),
            "Wrong-Way": ("—", "Requires tracking vectors", "PENDING", "red"),
            "Emergency Vehicles": (str(sum(counts.get(k, 0) for k in ("ambulance", "fire truck", "police"))), "Latest inference", "LIVE", "green"),
        }
    else:
        values = DEMO_STATS
        html('<div class="demo-note" style="margin-bottom:.75rem">DEMO TELEMETRY · Run an image or video inference to replace these illustrative command-center values with model-derived results.</div>')
    icons = ["◈", "◉", "▣", "▤", "△", "≋", "↔", "!"]
    cols = st.columns(4)
    for index, (label, (value, meta, change, tone)) in enumerate(values.items()):
        with cols[index % 4]:
            metric_card(label, value, meta, change, icons[index], tone)
        if index % 4 == 3 and index < 7:
            st.write("")


def render_intelligence() -> None:
    violations = [
        ("Helmet Violation", "43", "HIGH", "02:14:08", "Monitoring", "tag-high"),
        ("Triple Riding", "17", "MEDIUM", "02:11:32", "Review queue", "tag-medium"),
        ("Wrong-Way Driving", "09", "HIGH", "02:08:51", "Alert active", "tag-high"),
        ("Signal Violation", "28", "MEDIUM", "02:05:17", "Monitoring", "tag-medium"),
        ("Restricted Lane", "12", "LOW", "01:58:44", "Resolved", "tag-low"),
        ("Number Plate Detection", "86", "LOW", "02:13:59", "Indexed", "tag-low"),
    ]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Traffic Safety Intelligence", "Live classification of road-risk signals · demo stream")
    for name, count, severity, latest, state, tag in violations:
        html(f'<div class="mini-stat"><span>{name}</span><span><strong>{count}</strong>&nbsp;&nbsp; <span class="tag {tag}">{severity}</span>&nbsp;&nbsp; <span style="color:#6f8795">{latest}</span>&nbsp;&nbsp; <span class="status-row status-ok">{status_dot()} {state}</span></span></div>')
    st.markdown("</div>", unsafe_allow_html=True)


def render_emergency() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Emergency Response System", "Priority routing is simulated until a control-room integration is configured")
    html('<div style="display:flex;align-items:center;gap:1rem;padding:.9rem;border-radius:10px;background:#3d182233;border:1px solid #ff647a55;margin-bottom:1rem"><div style="font-size:1.55rem;color:#ff7183">!</div><div><div style="color:#ff91a0;font-weight:800;font-size:.8rem">EMERGENCY VEHICLE DETECTED</div><div style="color:#b98b93;font-size:.67rem;margin-top:.2rem">Latest event · EV-0194 · Ambulance</div></div><span class="tag tag-high" style="margin-left:auto">PRIORITY P1</span></div>')
    for label, value in [("Vehicle type", "Ambulance"), ("Confidence", "94.2%"), ("Current location", "North Junction / Camera 04"), ("Detection time", "02:12:46"), ("Tracking status", "Active · 0.8 km"), ("Priority level", "P1 — Critical")]:
        html(f'<div class="mini-stat"><span>{label}</span><strong>{value}</strong></div>')
    html('<div style="display:flex;gap:.5rem;margin-top:1rem"><span class="tag tag-medium">TRAFFIC CONTROL ALERT SENT*</span><span class="tag tag-low">PRIORITY ROUTE ACTIVATED*</span></div><div style="color:#718797;font-size:.61rem;margin-top:.6rem">* Simulated dashboard event. No police, emergency service, or external notification has been contacted.</div>')
    st.markdown("</div>", unsafe_allow_html=True)


def render_plate_table() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("License Plate Intelligence", "Masked/sample identifiers shown when confirmed plate data is unavailable")
    st.dataframe(
        [
            {"Vehicle": "VH-2048", "Plate": "MH 12 •• 4821", "Type": "Car", "Violation": "None", "Confidence": "96.8%", "Time": "02:14:02"},
            {"Vehicle": "VH-2037", "Plate": "KA 05 •• 1184", "Type": "Motorcycle", "Violation": "No helmet", "Confidence": "91.4%", "Time": "02:13:41"},
            {"Vehicle": "VH-2019", "Plate": "DL 01 •• 7702", "Type": "Truck", "Violation": "Restricted lane", "Confidence": "88.9%", "Time": "02:12:57"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_tracking() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Live Vehicle Tracking", "Trajectory data becomes available when video inference and tracking are enabled")
    st.dataframe(
        [
            {"Vehicle ID": "VH-2048", "Type": "Car", "Location": "North Junction", "Direction": "East →", "Speed": "42 km/h", "Status": "Tracked"},
            {"Vehicle ID": "VH-2037", "Type": "Motorcycle", "Location": "North Junction", "Direction": "South ↓", "Speed": "31 km/h", "Status": "Tracked"},
            {"Vehicle ID": "EV-0194", "Type": "Ambulance", "Location": "Sector 07", "Direction": "West ←", "Speed": "54 km/h", "Status": "Priority"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:.8rem">', unsafe_allow_html=True)
    panel_heading(
        "Detection Location",
        "Location information for the current video or webcam source",
    )

    location_rows = [
        ("GPS Status", "Not Available"),
        ("Location Source", "Video / Webcam"),
        ("Coordinates", "Not available"),
        ("Tracking", "Local detection only"),
    ]

    for label, value in location_rows:
        html(
            f'<div class="mini-stat">'
            f'<span>{label}</span>'
            f'<strong>{value}</strong>'
            f'</div>'
        )

    html(
        '<div class="demo-note" style="margin-top:.8rem">'
        'Location data is not available because the current video/webcam input '
        'does not provide GPS metadata. Detection remains local to the device.'
        '</div>'
    )
    st.markdown("</div>", unsafe_allow_html=True)


def make_charts() -> None:
    if px is None:
        st.info("Plotly is not installed. Install it to enable interactive analytics.")
        st.bar_chart({"Vehicles": [180, 240, 320, 410, 360, 290, 220]}, height=260)
        return
    hours = [f"{h:02d}:00" for h in range(6, 18)]
    vehicles = [108, 142, 187, 221, 248, 264, 250, 238, 272, 318, 301, 284]
    fig = px.area(x=hours, y=vehicles, labels={"x": "", "y": "Vehicles"}, template="plotly_dark")
    fig.update_traces(line_color="#4ddfe4", fillcolor="rgba(67,217,223,.15)")
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8da2b0")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_analytics() -> None:
    html('<div class="section"><div><h2>Analytics Command Desk</h2><p>Traffic patterns, violation density and model performance at a glance.</p></div><span class="eyebrow">ROLLING WINDOW / 12H</span></div>')
    kpi_cols = st.columns(4)
    for col, label, value, detail in [
        (kpi_cols[0], "Traffic volume", "6,284", "+9.4% vs yesterday"),
        (kpi_cols[1], "Safety score", "87.6", "+4.1 pts this week"),
        (kpi_cols[2], "Average confidence", "91.8%", "Across active models"),
        (kpi_cols[3], "Incidents resolved", "94%", "Within target SLA"),
    ]:
        with col:
            html(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-meta">{detail}</div></div>')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Traffic Analytics", "Rolling 12-hour operational view · demonstration telemetry")
    make_charts()
    c1, c2 = st.columns(2)
    with c1:
        if px:
            fig = px.bar(x=["Helmet", "Signal", "Wrong-way", "Lane"], y=[43, 28, 9, 12], labels={"x": "", "y": "Events"}, template="plotly_dark", color_discrete_sequence=["#5a8cff"])
            fig.update_layout(height=240, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8da2b0")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.bar_chart({"Helmet": [43], "Signal": [28], "Wrong-way": [9], "Lane": [12]})
    with c2:
        st.markdown("**Vehicle-type distribution**")
        st.progress(.67, text="Cars · 67%")
        st.progress(.24, text="Motorcycles · 24%")
        st.progress(.09, text="Heavy vehicles · 9%")
        html('<div class="mini-stat"><span>Helmet compliance</span><strong style="color:#5de0bd">93.7%</strong></div><div class="mini-stat"><span>Daily traffic volume</span><strong>6,284</strong></div><div class="mini-stat"><span>Emergency events</span><strong style="color:#f5c36c">06</strong></div>')
    st.markdown("</div>", unsafe_allow_html=True)


def render_incidents() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_heading("Incident Center", "Review, triage and export safety events from the monitoring stream")
    c1, c2, c3 = st.columns(3)
    c1.date_input("Date", value=datetime.now().date(), key="incident_date")
    c2.selectbox("Violation type", ["All violations", "Helmet violation", "Wrong-way driving", "Signal violation"], key="incident_violation")
    c3.selectbox("Severity", ["All severities", "High", "Medium", "Low"], key="incident_severity")
    st.dataframe(
        [
            {"Incident ID": "INC-2407", "Violation": "Helmet violation", "Vehicle": "VH-2037", "Timestamp": "07 Oct · 02:14", "Severity": "High", "Evidence": "Available", "Status": "Needs review"},
            {"Incident ID": "INC-2406", "Violation": "Wrong-way driving", "Vehicle": "VH-1988", "Timestamp": "07 Oct · 02:08", "Severity": "High", "Evidence": "Available", "Status": "Escalated"},
            {"Incident ID": "INC-2405", "Violation": "Restricted lane", "Vehicle": "VH-2019", "Timestamp": "07 Oct · 01:58", "Severity": "Low", "Evidence": "Available", "Status": "Resolved"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_health() -> None:
    st.markdown('<div class="section"><div><h2>AI System Health</h2><p>Operational readiness across the SAFE VISION stack</p></div></div>', unsafe_allow_html=True)
    items = [
        ("YOLO Model", "Online"),
        ("Camera Feed", "Ready"),
        ("Video Processing", "Active"),
        ("Detection Engine", "Healthy"),
        ("Database", "Demo"),
        ("Alert System", "Disabled"),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, items):
        with col:
            html(f'<div class="health-card"><span class="health-dot"></span><div><b>{label}</b><span style="color:#59dcbd">{value}</span></div></div>')


def render_video_detection(model: Any | None, confidence: float) -> None:
    st.markdown('<div class="section"><div><h2>Video Detection Lab</h2><p>Process local road footage with the selected YOLO pipeline.</p></div></div>', unsafe_allow_html=True)
    video = st.file_uploader("Upload road video", type=["mp4", "avi", "mov", "mkv"], key="video_detection")
    if video is not None:
        st.video(video.getvalue())
        st.info("Video processing runs locally and writes annotated output under runs/dashboard_video_outputs.")
        if st.button("Process video", key="process_video"):
            if cv2 is None:
                st.error("OpenCV is required for video processing.")
                return
            if model is None:
                st.warning("Select a model with valid weights before processing.")
                return
            input_path: Path | None = None
            cap: Any | None = None
            writer: Any | None = None
            try:
                output_dir = ROOT / "runs" / "dashboard_video_outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=Path(video.name).suffix,
                    delete=False,
                ) as temp_input:
                    temp_input.write(video.getvalue())
                    input_path = Path(temp_input.name)
                cap = cv2.VideoCapture(str(input_path))
                if not cap.isOpened():
                    st.error("Could not open this video file.")
                    return
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if np.isfinite(fps) and fps > 0 else 20.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width <= 0 or height <= 0:
                    st.error("Video dimensions could not be read.")
                    return
                output_path = output_dir / f"{Path(video.name).stem}_annotated.mp4"
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    st.error("Could not create the annotated output video.")
                    return
                progress = st.progress(0, text="Preparing local inference…")
                preview = st.empty()
                frame_number = 0
                total_detections = 0
                counts: Counter[str] = Counter()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = predict(model, frame, confidence)
                    annotated = result.plot()
                    writer.write(annotated)
                    total_detections += len(result.boxes)
                    counts.update(class_name(model, box).lower() for box in result.boxes)
                    frame_number += 1
                    if frame_number == 1 or frame_number % 15 == 0:
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Live preview · frame {frame_number}", use_container_width=True)
                    if total_frames > 0:
                        progress.progress(min(frame_number / total_frames, 1.0), text=f"Processing frame {frame_number} of {total_frames}")
                if frame_number == 0:
                    st.error("No readable frames were found in this video.")
                    return
                progress.progress(1.0, text="Inference complete")
                st.success(f"Processed {frame_number:,} frames with {total_detections:,} total detections.")
                html(f'<div class="demo-note">LIVE RESULT · {", ".join(f"{key}: {value}" for key, value in counts.most_common(5)) or "No classes detected"}</div>')
                st.video(str(output_path))
                with output_path.open("rb") as output_file:
                    st.download_button("Download annotated video", output_file.read(), file_name=output_path.name, mime="video/mp4")
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                if input_path is not None and input_path.exists():
                    input_path.unlink(missing_ok=True)


def render_settings() -> tuple[str, float]:
    st.markdown('<div class="section"><div><h2>System Settings</h2><p>Configure inference behavior and monitoring preferences.</p></div></div>', unsafe_allow_html=True)
    model_name = st.selectbox("Active detection model", list(MODEL_PATHS), key="settings_model")
    confidence = st.slider("Confidence threshold", .10, .95, .30, .05, key="settings_confidence")
    st.checkbox("Use local inference only", value=True, disabled=True)
    st.checkbox("External alert integration", value=False, disabled=True)
    html('<div class="demo-note">External alerts are intentionally disabled. This prototype does not contact police, emergency services, or third-party notification providers.</div>')
    return model_name, confidence


def main() -> None:
    inject_theme()
    active_page = render_sidebar()
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = active_page
    model_name = st.session_state.get("settings_model", "Traffic Detection")
    confidence = st.session_state.get("settings_confidence", .30)
    model, model_error = model_for(model_name)
    render_header()

    if model_error:
        html(f'<div class="demo-note">DEMO MODE · {model_error}</div>')

    if active_page == "Command Center":
        html('<div class="section"><div><h2>Command Center</h2><p>Live network overview · Sector 07 · Monitoring mode: <b style="color:#d8eaef">Standard</b></p></div><span class="eyebrow">STREAM / 04 ACTIVE</span></div>')
        render_metrics()
        st.write("")
        left, right = st.columns([1.65, 1])
        with left:
            render_monitor(model, confidence)
            detection_input(model, confidence)
        with right:
            render_intelligence()
            render_emergency()
        render_health()
    elif active_page == "Video Detection":
        render_video_detection(model, confidence)
        render_monitor(model, confidence, active_page)
    elif active_page == "Live Camera":
        render_live_camera(model, confidence)
    elif active_page == "Traffic Violations":
        html('<div class="section"><div><h2>Traffic Violations</h2><p>Classification and escalation queue for road-safety events.</p></div></div>')
        render_intelligence()
    elif active_page == "Emergency Vehicle Detection":
        html('<div class="section"><div><h2>Emergency Vehicle Detection</h2><p>Priority detection and route coordination workspace.</p></div></div>')
        render_emergency()
    elif active_page == "Number Plate Recognition":
        html('<div class="section"><div><h2>Number Plate Recognition</h2><p>Masked plate indexing for privacy-aware vehicle intelligence.</p></div></div>')
        render_plate_table()
    elif active_page == "Analytics":
        render_analytics()
    elif active_page == "Vehicle Tracking":
        render_tracking()
    elif active_page == "Incident Reports":
        render_incidents()
    elif active_page == "System Settings":
        render_settings()
    html('<div class="footer"><span>SAFE VISION AI · See Risk. Detect Threats. Protect Lives.</span><span>Prototype telemetry · Human verification required</span></div>')


if __name__ == "__main__":
    main()