# EASD (Embedded Anxiety & Sleep Detection) - Presentation Guide

## 1. The Vision & Problem Statement
In today’s fast-paced world, stress and anxiety are silent epidemics that often go unnoticed until they manifest clinically. Off-the-shelf smartwatches provide very basic step-counting and heart rate monitoring, but they lack the clinical depth to accurately classify complex neurological states like high-stress or specific sleep stages in real-time. 

Our goal was to build a comprehensive, continuous monitoring system from the ground up—combining clinical-grade biometric sensors with an edge-deployed Machine Learning model to detect stress and classify sleep autonomously.

---

## 2. The Hardware Architecture
To solve this, we couldn't rely on just one data point. Stress is a multi-system response. We engineered a custom wearable sensor array consisting of three distinct modules wired into a central processing hub:

1. **Muscle Electromyography (EMG):** We use an Analog-to-Digital Converter (ADC) to read micro-volt electrical signals directly from the user's muscles. When a person is stressed, their muscles unknowingly exhibit micro-contractions.
2. **Kinematics (MPU6050 Accelerometer/Gyroscope):** This 6-axis motion sensor tracks physical unrest, fidgeting, and sleep tossing. 
3. **Photoplethysmography (MAX30102 PPG):** We utilize an infrared and red LED sensor to measure blood volume changes. This gives us not just Heart Rate (BPM), but crucial millisecond-level Heart Rate Variability (HRV) metrics.

We brought all of this together on a **Raspberry Pi**—acting as our edge compute device. This means data isn't sent to a slow, expensive cloud server; the heavy lifting is done right on the device in your room.

---

## 3. Real-Time Signal Processing
Sensors generate noisy, chaotic data. If you feed raw data to an AI, it fails. So, we built a highly optimized, multi-threaded signal processing pipeline.

- Operating at high frequencies (up to 200 times per second for the EMG), our software runs continuous background threads. 
- We use advanced digital filters—like low-pass filters to remove gravity from motion data, and band-pass filters to isolate muscle signals from electrical line noise. 
- Once the signal is clean, we extract exactly **10 statistical features** over rolling 30-second windows. For example, instead of just saying 'your heart rate is 80', we calculate sub-metrics like the `RMSSD`—the exact mathematical variance between heartbeats, which is a gold-standard indicator of the body's 'fight or flight' response.

---

## 4. Machine Learning & Algorithms
At the heart of the EASD system is our Machine Learning model. Here is exactly how it is built, trained, and utilized:

### 4.1. Data Preprocessing & Balancing
Before feeding data into the machine learning algorithm, it has to be cleaned and structured:
*   **WESAD Dataset:** We trained our predictive model using the WESAD (Wearable Stress and Affect Detection) dataset, a premier clinical dataset for physiological stress. The algorithm uses the 10 features our sensors feed it and compares them against thousands of clinical profiles.
*   **SMOTE Class Balancing:** In real-world data, there is way more "relaxed" data than "stressed" data. If you train an AI blindly, it learns to just guess "relaxed" to score highly. Our code uses an upsampling algorithm (similar to SMOTE) to artificially duplicate and balance the minority class so there is a strict 50/50 split between stressed and non-stressed samples during training.
*   **StandardScaler (Normalization):** Heart rate variations might be tiny fractions (e.g., `0.04`), while accelerometer features might be large numbers (e.g., `9.8`). The pipeline uses a `StandardScaler` which mathematically forces every metric to have a mean of 0 and a standard deviation of 1. This prevents big numbers from overpowering small ones.

### 4.2. The Algorithms (Ensemble Learning)
Instead of relying on a single AI algorithm, the training script stages a competition between powerful mathematical models:
1.  **RandomForestClassifier:** This builds hundreds of individual "Decision Trees" on random subsets of the data. Each tree looks at different features and casts a vote on whether the person is stressed.
2.  **GradientBoostingClassifier:** This builds trees sequentially. Every new tree specifically studies the mistakes made by the previous tree and tries to correct them.
3.  **VotingClassifier (Soft Voting Ensemble):** This is a meta-algorithm. It takes the probability outputs from the Random Forest and the Gradient Booster, averages them together, and outputs a highly confident blended prediction.

### 4.3. Training Validations
To ensure the AI doesn't just memorize the training data (Overfitting), it undergoes **Stratified 5-Fold Cross-Validation**. The script carves the dataset into 5 equal chunks, trains on 4, and is blindly tested on the 1 chunk it hasn't seen. Whichever algorithm wins the 5-fold cross-validation (based on max Accuracy and minimum false alarms/F1 Score) is selected.

### 4.4. Deployment Pipeline Structure
The script permanently packages the *StandardScaler* and the *Winning Model* together into a single Scikit-Learn `Pipeline` object and saves it (`anxiety_model_v2.joblib`). Because it's a pre-packaged joblib model, inference (making a prediction) takes fractions of a second on the Raspberry Pi.

---

## 5. The User Experience & Dashboard
We wanted the data to be perfectly intelligible for both doctors and end-users. 

We developed a beautiful, dark-mode web application using Python and Streamlit. Because it is hosted directly on the Raspberry Pi over the local network, you can open a browser on your phone or laptop and instantly see a **Live Dashboard**. 

This dashboard visualizes:
*   Real-time stress probabilities updating continuously.
*   Estimated sleep stages (Wake, Light Sleep, Deep Sleep, REM).
*   Live telemetry of muscle tension and heart rate metrics. 
*   An export function allowing physicians to download entire sessions as CSV files for clinical review.

---

## 6. Deployment & The Future
Deployment is entirely automated via Git. When we push updates to the repository, the Raspberry Pi pulls down the changes and launches both the sensor polling scripts and the web dashboard autonomously via auto-start configs.

**Summary:** We took raw, noisy hardware components, built a custom multi-threaded Python driver layer, routed that into a clinical Machine Learning model, and wrapped it in a premium UI. EASD is now fully functional, running entirely completely offline at the edge, and ready to provide clinical-grade insights into human well-being.
