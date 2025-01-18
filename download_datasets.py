import kagglehub
import os
import shutil
datasets_path="./Datasets"

# Kotki pieski
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset" )

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), datasets_path)

# losowe rzeczy
path = kagglehub.dataset_download("pankajkumar2002/random-image-sample-dataset")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), datasets_path)
