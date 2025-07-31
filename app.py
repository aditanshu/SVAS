import streamlit as st
import cv2
import torch
from utils.detection import load_model, predict_sequence, preprocess_frame
from services.call_service import make_call
from services.location_service import get_location
import tempfile
import numpy as np
import folium
from streamlit_folium import st_folium
import threading

# Page configuration
st.set_page_config(page_title="SVAS - Smart Violence Alert System", layout="wide", page_icon="🛡️")

# Header
st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: red;'>🛡️ Smart Violence Alert System (SVAS)</h1>
        <h4>Live Detection | Map Alert | Instant Call</h4>
        <hr style='border: 1px solid #f44336;'>
    </div>
""", unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.header("⚙️ Detection Settings")
mode = st.sidebar.radio("Select Mode", ["📷 Live Camera", "📤 Upload Video"])
st.sidebar.info("Use *Live Camera* for real-time surveillance or *Upload Video* for testing.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("violence_lstm_model.pth", device)

# State variables
sequence = []
call_status = "❌ Call Not Made"
violence_count = 0
location = None

# Display Map
def display_map(loc):
    if loc and isinstance(loc, (list, tuple)) and len(loc) == 2:
        try:
            m = folium.Map(location=loc, zoom_start=15)
            folium.Marker(loc, tooltip="📍 Incident Location", icon=folium.Icon(color="red")).add_to(m)
            st_folium(m, width=700, height=400)
        except Exception as e:
            st.warning(f"⚠️ Unable to render map: {e}")
    else:
        st.info("⚠️ No valid location data available.")

# Threaded call trigger
def threaded_call():
    global call_status, location
    try:
        make_call()
        location = get_location()
        call_status = "✅ Call Forwarded"
    except Exception as e:
        call_status = f"❌ Call Failed: {e}"

# Frontend UI
if mode == "📷 Live Camera":
    st.markdown("### 🎥 Live Camera Feed")

    start_detection = st.button("▶️ Start Live Detection")
    stframe = st.empty()
    status_box = st.empty()

    if start_detection:
        cap = cv2.VideoCapture(0)

        sequence = []
        call_triggered = False
        violence_count = 0
        call_status = "❌ Call Not Made"
        location = None

        while True:
            ret, frame = cap.read()
            if not ret:
                status_box.warning("🚫 Unable to access webcam or no frame captured.")
                break

            try:
                resized = cv2.resize(frame, (224, 224))
                tensor = preprocess_frame(resized)
                sequence.append(tensor)
            except Exception as e:
                status_box.warning(f"[Frame skipped] {e}")
                continue

            if len(sequence) == 20:
                pred = predict_sequence(model, sequence, device)
                sequence = []

                if pred == 1:
                    violence_count += 1
                    cv2.putText(frame, f"🚨 Violence Detected ({violence_count})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    status_box.error(f"🚨 Violence Detected! (Count: {violence_count})")

                    if not call_triggered:
                        threading.Thread(target=threaded_call).start()
                        call_triggered = True
                else:
                    cv2.putText(frame, "✅ No Violence", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    status_box.success("✅ No Violence Detected")

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

elif mode == "📤 Upload Video":
    st.markdown("### 📁 Upload a Video File")
    uploaded_file = st.file_uploader("Upload a video (.mp4 or .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        status_box = st.empty()

        sequence = []
        call_triggered = False
        violence_count = 0
        call_status = "❌ Call Not Made"
        location = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                resized = cv2.resize(frame, (224, 224))
                tensor = preprocess_frame(resized)
                sequence.append(tensor)
            except Exception as e:
                status_box.warning(f"[Frame skipped] {e}")
                continue

            if len(sequence) == 20:
                pred = predict_sequence(model, sequence, device)
                sequence = []

                if pred == 1:
                    violence_count += 1
                    status_box.error(f"🚨 Violence Detected! (Count: {violence_count})")

                    if not call_triggered:
                        threading.Thread(target=threaded_call).start()
                        call_triggered = True
                else:
                    status_box.success("✅ No Violence Detected")

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

# Incident Location Map
with st.expander("📍 View Incident Location on Map", expanded=True):
    if location:
        display_map(location)
    else:
        st.info("No incident location recorded yet.")

# Footer Stats
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 📞 Call Status:\n{call_status}")
with col2:
    st.markdown(f"### 🔢 Total Violence Incidents:\n{violence_count}")
