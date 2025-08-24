import os

base_dir = "data"
classes = ["Asthma", "Pneumonia", "Healthy"]

for label in classes:
    folder = os.path.join(base_dir, label)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    
    for i, file in enumerate(sorted(files), 1):
        new_name = f"{label.lower()}_{i:04d}.wav"
        old_path = os.path.join(folder, file)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)

print("✅ All files renamed successfully!")
