import kagglehub

# Download latest version
path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset", destination = "./Datasets" )

print("Path to dataset files:", path)