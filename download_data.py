import kagglehub

# Download latest version
path = kagglehub.dataset_download("dhinaharp/mushroom-dataset")

print("Path to dataset files:", path)