import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 500

rows = []

for _ in range(n_samples):
    label = np.random.choice([0, 1])  # 0 = relaxed, 1 = stressed
    
    if label == 0:
        emg_rms = np.random.normal(5e-5, 1e-5)
        emg_var = np.random.normal(2e-9, 5e-10)
        emg_mean = np.random.normal(0, 1e-6)
        acc_mean = np.random.normal(9.8, 0.05)
        acc_std = np.random.normal(0.05, 0.02)
        acc_max = np.random.normal(9.9, 0.1)
    else:
        emg_rms = np.random.normal(1.5e-4, 3e-5)
        emg_var = np.random.normal(6e-9, 1e-9)
        emg_mean = np.random.normal(0, 2e-6)
        acc_mean = np.random.normal(10.2, 0.2)
        acc_std = np.random.normal(0.2, 0.05)
        acc_max = np.random.normal(10.5, 0.3)

    rows.append([
        emg_rms,
        emg_var,
        emg_mean,
        acc_mean,
        acc_std,
        acc_max,
        label
    ])

columns = [
    "emg_rms",
    "emg_var",
    "emg_mean",
    "acc_mean",
    "acc_std",
    "acc_max",
    "label"
]

df = pd.DataFrame(rows, columns=columns)

os.makedirs("data", exist_ok=True)

file_path = "data/synthetic_anxiety_dataset.csv"
df.to_csv(file_path, index=False)

print("✅ Dataset created at:", file_path)
print(df.head())
