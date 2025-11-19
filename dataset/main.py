import os
import numpy as np
import pandas as pd

# -----------------------------
# Folder paths
# -----------------------------
base_paths = {
    "Healthy": ("EEG_data/Healthy/Eyes_open", 0),   # Alzheimer = 0
    "AD": ("EEG_data/AD/Eyes_open", 1)              # Alzheimer = 1
}

all_files = []

# -----------------------------
# Collect all files from Healthy + AD
# -----------------------------
for group, (base_path, label) in base_paths.items():
    if not os.path.exists(base_path):
        continue

    for patient in sorted(os.listdir(base_path)):
        patient_path = os.path.join(base_path, patient)
        if not os.path.isdir(patient_path):
            continue

        for f in sorted(os.listdir(patient_path)):
            if f.endswith(".txt"):
                file_path = os.path.join(patient_path, f)
                all_files.append((file_path, patient, label))

print(f"Total EEG files found: {len(all_files)}")

# -----------------------------
# Step 1: Find MINIMUM signal length across all files
# -----------------------------
lengths = [len(np.loadtxt(fp)) for fp, _, _ in all_files]
min_length = min(lengths)

print(f"✅ Using minimum signal length: {min_length}")

# -----------------------------
# Step 2: Build the dataset
# -----------------------------
rows = []

for file_path, patient, label in all_files:
    channel = os.path.basename(file_path).replace(".txt", "")
    signal = np.loadtxt(file_path)[:min_length]  # trim to min length

    row = {
        "Alzheimer": label,         # 0 = Healthy, 1 = AD
        "Patient": patient,
        "Channel": channel,
        "Eyes_State": "open"
    }

    # Add samples
    row.update({f"Sample_{i+1}": v for i, v in enumerate(signal)})
    rows.append(row)

# -----------------------------
# Step 3: Convert to DataFrame
# -----------------------------
df = pd.DataFrame(rows)

print("Final dataset shape:", df.shape)
print(df.head())

# -----------------------------
# Step 4: Save to CSV
# -----------------------------
df.to_csv("combined_eyes_open_dataset.csv", index=False)
print("💾 Saved as combined_eyes_open_dataset.csv")
