Here’s the updated `README.md` content with proper Markdown heading tags for better visibility:

---

````markdown
# 🛡️ Smart Violence Alert System (SVAS)

The **Smart Violence Alert System (SVAS)** is an AI-powered application that detects violent activity from real-time or recorded CCTV footage. It integrates a deep learning model with a user-friendly interface and geolocation-based alert system to assist emergency response teams in identifying and responding to potential violent incidents effectively.

---

## 🚀 Features

- 🎥 **Real-time & Offline Video Analysis**  
  Detects violence in both live camera feeds and uploaded videos.

- 🧠 **Deep Learning-Based Detection**  
  Utilizes a CNN-LSTM model trained on the *Real-Life Violence Situations* dataset.

- 🗺️ **Location Mapping**  
  Automatically displays the incident location on a map using Google Maps API.

- 📞 **Auto Alert System**  
  Triggers an emergency call to a predefined control room number upon violence detection (simulated via browser for demo).

- 📊 **Heatmap Visualization**  
  Displays incident trends and density with dynamic red-dot visualizations on the map.

---

## 🧠 Model Architecture

- **Feature Extractor**: Pretrained `ResNet18` (excluding FC layers).
- **Temporal Analysis**: LSTM layer processes features over frames.
- **Classifier**: Fully connected layers for final binary classification (Violence / Non-Violence).
- **Data**: [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (3.5GB)

---

## 📂 Project Structure

```bash
violence-detection-system/
│
├── flask_app/                 # Frontend using Flask + HTML + JS + Google Maps
│   ├── templates/
│   ├── static/
│   └── app.py
│
├── streamlit_app/             # Optional Streamlit interface
│
├── models/                    # Deep learning models
│   └── model_def.py
│
├── utils/                     # Utility functions
│   ├── feature_extractor.py
│   ├── detection.py
│   └── location.py
│
├── saved_features/            # Cached frame features for faster training
│
├── videos/                    # Input videos for testing
│
├── training/                  # Training and evaluation notebooks
│   └── violence_detection.ipynb
│
├── requirements.txt
└── README.md
````

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/violence-detection-system.git
cd violence-detection-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
Extract it inside the project’s `videos/` or `dataset/` folder.

---

## ▶️ Running the App

### 📦 Option 1: Flask Web Interface

```bash
cd flask_app
python app.py
```

Access it at `http://localhost:5000`

### 📊 Option 2: Streamlit Interface

```bash
python -m streamlit run app.py
```

---

## 🧪 Training the Model

Use the provided notebook:

```bash
training/violence_detection.ipynb
```

Supports frame-wise feature extraction + LSTM training using cached `.npy` files for speed.

---

## 📍 Output Preview

* ✅ Prediction: `Violence` or `Non-Violence`
* 📍 Location shown on live map
* 🔴 Heatmap builds over time
* 📱 Call triggered to control center (demo)

---

## 📌 Future Work

* Integrate SMS/email alerts
* Build mobile app with live video input
* Connect to actual police control room systems
* Add sound/event trigger detection (e.g., gunshot, scream)

---

## 🤝 Contributors

* **Aditanshu Bharadwaj Mishra** – Developer, Model Designer, System Integrator
* **OpenAI ChatGPT** – Assistant for architecture, codebase, and documentation

---

## 📜 License

This project is licensed under the MIT License.

---