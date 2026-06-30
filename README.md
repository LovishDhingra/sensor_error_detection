
# 🔧 Sensor Fault Detection System

An end-to-end **Machine Learning–based Sensor Fault Detection System** designed to automatically identify faulty and non-faulty sensor readings across multiple sensor types using a unified architecture.  
The system supports **Wafer, Gas, Temperature, Light (LDR), and Soil Moisture sensors** and provides real-time predictions through a Streamlit web interface.

🔗 **Live Demo:** https://sensor-fault-detection-system.streamlit.app/

---

## 📌 Project Highlights

- ✅ Multi-sensor fault detection using Machine Learning  
- ✅ All-in-One intelligent router for automatic sensor type detection  
- ✅ Robust preprocessing with imputation, scaling, and class balancing  
- ✅ Separate optimized models for each sensor type  
- ✅ Interactive Streamlit-based web interface  
- ✅ Production-ready modular project structure  

---

## 🧠 System Architecture

1. **Input Layer**  
   - CSV file upload or manual sensor input  

2. **All-in-One Router**  
   - Automatically detects sensor type based on schema  

3. **Preprocessing Pipeline**  
   - Missing value handling (KNN / Simple Imputer)  
   - Feature scaling (RobustScaler / StandardScaler)  
   - Class imbalance correction (SMOTETomek)  
   - Feature engineering  

4. **ML Model Layer**  
   - Wafer → Support Vector Classifier (SVC)  
   - Gas → Random Forest  
   - Temperature → Random Forest  
   - Light (LDR) → XGBoost  
   - Soil Moisture → XGBoost  

5. **Prediction & Visualization**  
   - Fault classification  
   - Downloadable results  
   - Visual insights  

---

## 📂 Supported Sensors

| Sensor Type | Fault Labels |
|------------|--------------|
| Wafer | 0 = Normal, 1 = Faulty |
| Gas | 0 = Normal, 1 = Faulty |
| Temperature | 0 = Faulty, 1 = Normal |
| Light (LDR) | -1 = Faulty, 1 = Normal |
| Soil Moisture | -1 = Faulty, 1 = Normal |

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Data Processing:** NumPy, Pandas  
- **Imbalance Handling:** SMOTETomek  
- **Model Persistence:** Pickle (.pkl)  
- **Web Framework:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

```
Sensor-Fault-Detection/
│
├── data/                   # Sensor datasets
├── models/                 # Trained models (.pkl)
├── pipelines/              # Preprocessing pipelines
├── scripts/                # Training & utility scripts
├── custom_transformers.py  # Feature engineering logic
├── model_utils.py          # Model load/save utilities
├── all_in_one_router.py    # Sensor auto-detection logic
├── streamlit_app.py        # Web application
├── main.py                 # Training & prediction entry
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/debjithacks/sensor-fault-detection.git
cd sensor-fault-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Evaluation

The app can now compute real evaluation metrics for you - no notebook required.

### Option A: In the web app

If your uploaded CSV includes the **actual/ground-truth fault status** (not just raw sensor readings), a **"📊 Model Evaluation Metrics"** section appears under your predictions:

1. Pick which column holds the ground-truth label.
2. Confirm which raw value means *Faulty* in that column, and which value the model uses for *Faulty* in its predictions (the app guesses a sensible default for both, but always lets you double-check/override - see the note below on `light`/LDR).
3. Instantly see **Accuracy, Precision, Recall, F1-score, Specificity, Balanced Accuracy, Matthews Correlation Coefficient, Cohen's Kappa, ROC-AUC, PR-AUC**, a **confusion matrix**, **ROC/PR curves**, and a **per-class report** - plus a downloadable `.txt` report.

### Option B: Command line (`evaluate_model.py`)

For batch evaluation against a labelled test CSV, without opening the UI:

```bash
python evaluate_model.py --sensor gas --data test_gas.csv --label-col faulty
```

This prints the full metrics report to the terminal and writes a `predictions.csv`, `metrics_report.txt`, `metrics_summary.json`, `confusion_matrix.png`, and (when probabilities are available) `roc_curve.png` / `pr_curve.png` to `evaluation_report/` (override with `--output-dir`). Run `python evaluate_model.py --help` for all options, including `--fault-value` / `--pred-fault-value` to override the auto-detected label encoding.

### A note on label encoding

This README documents the *raw dataset* convention:

| Sensor | Fault Labels |
|------------|--------------|
| Wafer | 0 = Normal, 1 = Faulty |
| Gas | 0 = Normal, 1 = Faulty |
| Temperature | 0 = Faulty, 1 = Normal |
| Light (LDR) | -1 = Faulty, 1 = Normal |
| Soil Moisture | -1 = Faulty, 1 = Normal |

While building the evaluation feature we found that the **deployed `ldr_pipeline.joblib`** actually outputs `0`/`1` (not `-1`/`1`) - whatever remapping happened, it happened before training and isn't reversed at inference time, so `model.predict()` never returns `-1`. We *assume* `0` corresponds to the original `-1` (Faulty), matching scikit-learn's default ascending `LabelEncoder` order, but this hasn't been verified against the original training notebook. The Soil-Moisture pipeline, by contrast, does decode back to `-1`/`1` correctly. Both the web app and `evaluate_model.py` always show you the actual values present and let you pick/override which one means "Faulty" rather than trusting this silently - if you have the original training code, it's worth double-checking the LDR convention and updating `evaluation_utils.SENSOR_LABEL_INFO` if needed.

---

## 👥 Team Members

This project was developed as a **team project** by:

- Aditya Maity  
- Souvik Pramanik  
- Avijit Ray  
- Subhadeep Hatai  
- **Debjit Ghosh**  

---

## 📌 Use Cases

- Industrial sensor health monitoring  
- Smart agriculture systems  
- IoT-based automation platforms  
- Preventive maintenance systems  

---

## 🔮 Future Enhancements

- Real-time sensor streaming (IoT integration)
- Deep learning-based fault detection
- Sensor fusion and anomaly explanation
- Cloud-native deployment

---

## 📜 License

This project is released under the **MIT License** – free to use, modify, and distribute with attribution.

---

⭐ If you find this project useful, please consider giving it a star!
