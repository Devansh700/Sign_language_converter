# train_model.py
# Train an ASL alphabet classifier using MobileNetV2

import os
import json
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# -------- CONFIG --------
# Path detection logic
dataset_path_file = "dataset_path.txt"
if os.path.exists(dataset_path_file):
    with open(dataset_path_file, "r") as f:
        extract_folder = f.read().strip()
else:
    extract_folder = "dataset" # default fallback

print(f"[INFO] Using dataset from: {extract_folder}")

img_size = (224, 224)
batch_size = 32
epochs = 5  # Reduced epochs for faster testing
os.makedirs("models", exist_ok=True)
model_save_path = os.path.join("models", "asl_model.keras")
classes_save_path = "classes.json"
# -------------------------

# -------- DATASET --------
train_ds = tf.keras.utils.image_dataset_from_directory(
    extract_folder,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    extract_folder,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Classes
classes = train_ds.class_names
print("[OK] Classes found:", classes)

# Save classes to JSON
with open(classes_save_path, "w") as f:
    json.dump(classes, f)
print(f"[OK] Classes saved to {classes_save_path}")

num_classes = len(classes)

# Normalize [0,1] and prefetch for performance
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

# -------- MODEL --------
def build_model(input_shape=(224,224,3), num_classes=26):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False   # freeze base layers initially

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base.input, outputs=outputs)

model = build_model((*img_size, 3), num_classes)
model.compile(optimizer=Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------- CALLBACKS --------
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy',
                             save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_accuracy', patience=4,
                      restore_best_weights=True, verbose=1)

# -------- TRAIN --------
print("[START] Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early]
)

print("[DONE] Training complete. Best model saved at:", model_save_path)
print("[DONE] Classes saved at:", classes_save_path)
