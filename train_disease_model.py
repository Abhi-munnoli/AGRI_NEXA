import os
import json
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =====================================================
# YIELDNEXT - AI CROP DISEASE MODEL TRAINING
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "disease_dataset")
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "disease_classes.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
VALIDATION_SPLIT = 0.20
SEED = 42
FINE_TUNE_LAYERS = 40

print("\n========================================")
print("       YIELDNEXT DISEASE MODEL")
print("========================================")

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"disease_dataset folder not found:\n{DATASET_DIR}"
    )

# =====================================================
# LOAD DATA
# =====================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\n========================================")
print("DISEASE CLASSES")
print("========================================")
print("Total classes:", num_classes)

for i, name in enumerate(class_names):
    print(f"{i:02d} -> {name}")

if num_classes != 38:
    print(
        f"\nWARNING: Expected 38 classes, but found {num_classes}."
    )

with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        class_names,
        f,
        indent=4,
        ensure_ascii=False
    )

# =====================================================
# PERFORMANCE
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# =====================================================
# DATA AUGMENTATION
# =====================================================

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# =====================================================
# MOBILENETV2
# =====================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================
# Training preprocessing is [-1, 1].
# app.py must use MobileNetV2 preprocess_input().
# =====================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="leaf_image"
)

x = data_augmentation(inputs)

x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1,
    name="mobilenetv2_preprocessing"
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="disease_predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="YieldNext_Disease_MobileNetV2"
)

# =====================================================
# CALLBACKS
# =====================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr
]

# =====================================================
# STAGE 1
# =====================================================

print("\n========================================")
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("========================================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks
)

# =====================================================
# STAGE 2 - FINE TUNING
# =====================================================

print("\n========================================")
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("========================================")

base_model.trainable = True

fine_tune_start = max(
    0,
    len(base_model.layers) - FINE_TUNE_LAYERS
)

for layer in base_model.layers[:fine_tune_start]:
    layer.trainable = False

# Keep BatchNormalization frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print(
    "MobileNetV2 trainable layers:",
    trainable_count
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# =====================================================
# LOAD BEST MODEL
# =====================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model.save(MODEL_PATH)

# =====================================================
# FINAL EVALUATION
# =====================================================

print("\n========================================")
print("FINAL EVALUATION")
print("========================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print(
    f"\nFinal validation accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Final validation loss: {loss:.4f}"
)

# =====================================================
# OUTPUT CHECK
# =====================================================

for images, labels in validation_dataset.take(1):
    predictions = model.predict(
        images,
        verbose=0
    )

    print("\n========================================")
    print("MODEL OUTPUT CHECK")
    print("========================================")
    print("Input batch shape :", images.shape)
    print("Prediction shape  :", predictions.shape)
    print("Expected classes  :", num_classes)

    if predictions.shape[-1] != num_classes:
        raise RuntimeError(
            "Model output count does not match class count."
        )

    print("Output verification: PASSED")

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")
print("Model saved   :", MODEL_PATH)
print("Classes saved :", CLASS_PATH)
print("\nYieldNext Disease Detection is READY!")
