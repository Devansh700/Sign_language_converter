import kagglehub
import os
import shutil

# 1. Download dataset using kagglehub
print("Downloading ASL Alphabet dataset from Kaggle...")
download_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print(f"Downloaded to: {download_path}")

# 2. Check structure and set paths
train_data_source = os.path.join(download_path, "asl_alphabet_train", "asl_alphabet_train")

if not os.path.exists(train_data_source):
    print("Checking alternative structure...")
    train_data_source = os.path.join(download_path, "asl_alphabet_train")

# 3. Create a local reference
local_dataset_path = "dataset"
if os.path.exists(local_dataset_path):
    if os.path.islink(local_dataset_path):
        os.unlink(local_dataset_path)
    else:
        shutil.rmtree(local_dataset_path)

print(f"Linking dataset to {local_dataset_path}...")
try:
    with open("dataset_path.txt", "w") as f:
        f.write(train_data_source)
    print(f"Path saved to dataset_path.txt: {train_data_source}")
except Exception as e:
    print(f"Error: {e}")

print("\nSetup complete! Now you can run: python train_model.py")
