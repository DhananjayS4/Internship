# 🧠 EASD — Embedded Anxiety & Sleep Detection System

Real-time stress and sleep stage monitoring using EMG, accelerometer, PPG and a Random Forest ensemble trained on the **WESAD** dataset.

---

## Hardware (Raspberry Pi)
| Sensor | Purpose | Interface |
|--------|---------|-----------|
| ADS1115 | EMG acquisition | I²C |
| MPU-6050 | 3-axis accelerometer | I²C |
| MAX30102 | PPG / heart rate (SpO₂) | I²C |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# On Raspberry Pi, also install:
pip install smbus2 adafruit-circuitpython-ads1x15 mpu6050-raspberrypi
```

### 2. Download & place WESAD dataset
Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/

Place subject folders under:
```
data/WESAD/WESAD/S2/S2.pkl
data/WESAD/WESAD/S3/S3.pkl
...
data/WESAD/WESAD/S17/S17.pkl
```

### 3. Build WESAD feature dataset  *(~10–20 min, heavy CPU)*
```bash
python ml/build_wesad_dataset.py
```
Output: `data/wesad_features.csv`

### 4. Train the advanced model
```bash
python ml/train_advanced.py
```
Output: `ml/models/anxiety_model_v2.joblib` + `data/model_metrics.json`

### 5. Launch the dashboard
```bash
streamlit run ui/dashboard.py
```

### 6. Real-time loop (headless, on Pi)
```bash
python main.py
```

---

## Project Structure
```
EASD_System/
├── main.py                    Real-time prediction loop
├── sensor_service.py          Unified sensor class
├── drivers/                   Hardware abstraction layer
│   ├── emg_adc.py             ADS1115 EMG
│   ├── imu_mpu6050.py         MPU-6050 accelerometer
│   └── max30102_driver.py     MAX30102 PPG
├── processing/
│   ├── filtering.py           Butterworth filters
│   ├── features.py            Feature extraction (EMG, Accel, HRV, EDA)
│   ├── windowing.py           SlidingWindow
│   └── logger.py              CSV logger
├── ml/
│   ├── wesad_loader.py        WESAD .pkl loader
│   ├── wesad_features.py      Windowed feature extractor
│   ├── build_wesad_dataset.py Build training CSV from WESAD
│   ├── train_advanced.py      Ensemble training with 5-fold CV
│   ├── build_dataset.py       Synthetic dataset generator (legacy)
│   ├── train.py               Simple RF trainer (legacy)
│   └── models/
│       ├── anxiety_model.joblib      v1 (synthetic)
│       └── anxiety_model_v2.joblib   v2 (WESAD)
├── ui/
│   └── dashboard.py           Premium Streamlit dashboard
├── data/
│   ├── synthetic_anxiety_dataset.csv
│   ├── wesad_features.csv     (generated)
│   └── model_metrics.json     (generated)
└── requirements.txt
```

---

## Features Used
| Feature | Source | Description |
|---------|--------|-------------|
| `emg_rms` | Chest EMG | Root mean square of EMG signal |
| `emg_var` | Chest EMG | Signal variance |
| `emg_mean` | Chest EMG | Mean absolute amplitude |
| `acc_mean` | Chest ACC | Mean acceleration magnitude |
| `acc_std` | Chest ACC | Std of acceleration magnitude |
| `acc_max` | Chest ACC | Max acceleration magnitude |
| `hrv_rmssd` | Wrist BVP | Root mean square successive RR differences |
| `hrv_sdnn` | Wrist BVP | Std of RR intervals |
| `hrv_pnn50` | Wrist BVP | % successive RR > 50ms |
| `eda_mean` | Chest EDA | Mean skin conductance |

---

## Model Performance (WESAD)
See `data/model_metrics.json` after training. Expected accuracy ≥ 85%.
