# train.py
# Train an ASL alphabet classifier using MobileNetV2 (TF-Keras 3 compatible)

import os
import json
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# -------- CONFIG --------
extract_folder = r"D:\ME\Major Project\asl_alphabet_test"     # dataset folder (A, B, C, ...)
img_size = (224, 224)
batch_size = 32
epochs = 12
model_save_path = r"D:\ME\Major Project\Save\asl_model.keras"
classes_save_path = r"D:\ME\Major Project\classes.json"  # file where classes will be saved
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

# 1️⃣ Classes (prefetch/map se pehle)
classes = train_ds.class_names
print("✅ Classes found:", classes)

# Save classes to JSON
with open(classes_save_path, "w") as f:
    json.dump(classes, f)
print(f"✅ Classes saved to {classes_save_path}")

num_classes = len(classes)

# 2️⃣ Normalize [0,1] and prefetch for performance
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
print("🚀 Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early]
)

print("🎉 Training complete. Best model saved at:", model_save_path)
print("📂 Classes saved at:", classes_save_path)
