import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Paths
# -----------------------------
train_dir = r"D:\virtual_interview_simulator\AAA\ALL\train"
test_dir  = r"D:\virtual_interview_simulator\AAA\ALL\test"

img_size = 224   # ResNet expects 224x224
batch_size = 32

# -----------------------------
# 1. Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# 2. Build Optimized Model (ResNet50 Transfer Learning)
# -----------------------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# -----------------------------
# 3. Training with Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_emotion_model.keras", monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=callbacks
)

# -----------------------------
# 4. Evaluate
# -----------------------------
loss, acc = model.evaluate(test_generator, verbose=0)
print(f"✅ Final Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# 5. Save Final Model
# -----------------------------
model.save("final_emotion_model.keras")
print("Model saved as final_emotion_model.keras")
