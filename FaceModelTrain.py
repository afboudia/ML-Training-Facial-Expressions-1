import tensorflow as tf
import collections
from tensorflow.keras.layers import RandomFlip

# Dataset paths
train_dir = "fer2013/train"
test_dir = "fer2013/test"

# Image size & batch size
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32

# Datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale"
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale"
)

# Weighs the classes
class_names = train_dataset.class_names
num_classes = len(class_names)

train_labels = []
for _, labels in train_dataset:
    train_labels.extend(labels.numpy())

counter = collections.Counter(train_labels)
total_samples = sum(counter.values())
class_weights = {i: total_samples / (len(counter) * counter[i]) for i in counter.keys()}

# Train for flipping at random
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Normalize pixel values (0-255 → 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train with Class Weights to Handle Imbalance
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    class_weight=class_weights
)

model.save("emotion_model.keras")
