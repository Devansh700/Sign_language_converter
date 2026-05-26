import zipfile
import os
import shutil

zip_file_path =  "D:\\help\\asl_alphabet_test.zip"

extract_folder = "D:\ME\Major Project"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Extracted al files to {extract_folder}")

dataset_path = os.path.join(extract_folder, "asl_alphabet_test")



for file in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, file)
    if os.path.isfile(file_path):
        letter = file[0].upper()  # filename se letter extract karo
        letter_folder = os.path.join(dataset_path, letter)
        os.makedirs(letter_folder, exist_ok=True)
        shutil.move(file_path, os.path.join(letter_folder, file))
        
print(f"Dataset path set to: {dataset_path}")



