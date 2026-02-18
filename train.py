import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Configuration (FAST VERSION)
# ----------------------------

dataset_path = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5   # 🔥 Reduced from 20 → 5

# ----------------------------
# Data Preprocessing
# ----------------------------

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ----------------------------
# Load Pretrained Model
# ----------------------------

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Transfer Learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:\n")
model.summary()

# ----------------------------
# Callbacks (Optimized)
# ----------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,   # 🔥 Stop early if not improving
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_face_mask_model.keras",  # 🔥 Updated format
    monitor='val_accuracy',
    save_best_only=True
)

# ----------------------------
# Train Model
# ----------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ----------------------------
# Evaluation
# ----------------------------

val_generator.reset()
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype(int)

cm = confusion_matrix(val_generator.classes, predicted_classes)

print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(val_generator.classes, predicted_classes))

# ----------------------------
# Save Final Model
# ----------------------------

model.save("face_mask_model.keras")

print("\n✅ Model Training Complete & Saved Successfully!")
