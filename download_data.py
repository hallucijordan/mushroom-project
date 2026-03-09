import kagglehub
import os

# Download dataset into project's dt/ directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dt")
os.makedirs(output_dir, exist_ok=True)

path = kagglehub.dataset_download("dhinaharp/mushroom-dataset", output_dir=output_dir)

print("Path to dataset files:", path)