import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import numpy as np

dataset_path = "dataset"

# Supported image formats
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# ----------------------------
# 1️⃣ Class Distribution
# ----------------------------

classes = os.listdir(dataset_path)
data = []

for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    
    # Count only valid image files
    images = [f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTENSIONS)]
    
    data.append([cls, len(images)])

df = pd.DataFrame(data, columns=["Class", "Image_Count"])

print("\n📊 Dataset Summary:")
print(df)

plt.figure(figsize=(6,5))
sns.barplot(x="Class", y="Image_Count", data=df)
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.show()

# ----------------------------
# 2️⃣ Show Sample Images
# ----------------------------

plt.figure(figsize=(10,5))

for i, cls in enumerate(classes):
    class_path = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTENSIONS)]
    
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:  # Skip corrupt images
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, 2, i+1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            break  # show only first valid image

plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ Image Size Analysis
# ----------------------------

heights = []
widths = []
corrupt_count = 0

for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTENSIONS)]
    
    for img_name in images[:100]:  # check first 100 images
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            h, w, c = img.shape
            heights.append(h)
            widths.append(w)
        else:
            corrupt_count += 1

print("\n🖼 Sample Image Dimensions (first 5):")
print(list(zip(heights[:5], widths[:5])))

print("\n⚠ Corrupt Images Found During EDA:", corrupt_count)

# ----------------------------
# 4️⃣ Image Size Distribution
# ----------------------------

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(heights, bins=20)
plt.title("Image Height Distribution")

plt.subplot(1,2,2)
sns.histplot(widths, bins=20)
plt.title("Image Width Distribution")

plt.tight_layout()
plt.show()

# ----------------------------
# 5️⃣ Final Dataset Info
# ----------------------------

print("\n📌 Final Observations:")
print("Total Images:", sum(df["Image_Count"]))
print("Average Height:", int(np.mean(heights)))
print("Average Width:", int(np.mean(widths)))
