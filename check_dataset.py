import os

dataset_path = "dataset"

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)
    print(cls, "→", len(os.listdir(class_path)))
