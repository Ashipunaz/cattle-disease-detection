import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0

CLASS_NAMES = ['fmd', 'healthy', 'lumpy skin', 'mastitis']

data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.15),
    RandomContrast(0.2),
    RandomBrightness(0.2),
], name="data_augmentation")

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = data_augmentation(inputs)
x       = base_model(x, training=False)
x       = GlobalAveragePooling2D()(x)
x       = BatchNormalization()(x)
x       = Dropout(0.4)(x)
x       = Dense(512, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)

model = Model(inputs, outputs)
print("✅ Model built successfully")
print(f"Total params: {model.count_params()}")

model.load_weights('cattle_disease_model_v1.1.weights.h5')
print("✅ Weights loaded successfully")