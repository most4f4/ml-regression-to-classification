import warnings
warnings.filterwarnings('ignore')

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import Sequential, layers
from keras.optimizers import SGD
from keras.src.legacy.preprocessing.image import ImageDataGenerator



print("\n" + "="*50)
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("="*50)


# Load TRAINING data
train_data = []
train_labels = []


train_files = glob.glob('dataset/train/*/*')  
print(f"Found {len(train_files)} training files")

for i, address in enumerate(train_files):
    img = cv2.imread(address)

    if img is None:
        print(f"Failed to load image: {address}")
        continue  # skip this image


    img = cv2.resize(img, (32, 32))     # Resize the image to 32x32 pixels
    img = img / 255.0                   # Normalize the image between 0 and 1

    train_data.append(img)                    # Append the flattened image to the data list

    # address example: dataset/cat/cat.1.jpg
    label = address.split('\\')[-2]             # Get the label from the file path
    train_labels.append(label)                  # Append the label to the labels list

    if i % 500 == 0:
        print(f"[INFO] Processed {i} training images")

# Load TEST data
test_data = []
test_labels = []

test_files = glob.glob('dataset/test/*/*')  
print(f"Found {len(test_files)} test files")

for i, address in enumerate(test_files):
    img = cv2.imread(address)

    if img is None:
        print(f"Failed to load image: {address}")
        continue
    
    img = cv2.resize(img, (32, 32))
    img = img / 255.0

    test_data.append(img)

    # Handle both Windows and Unix path separators
    label = address.replace('\\', '/').split('/')[-2]
    test_labels.append(label)

    if i % 200 == 0:
        print(f"[INFO] Processed {i} test images")

# Convert to numpy arrays
X_train = np.array(train_data)
y_train = np.array(train_labels)
X_test = np.array(test_data)
y_test = np.array(test_labels)


# Encoding labels for CNN
le_cnn = LabelEncoder()
y_train_cnn = le_cnn.fit_transform(y_train)
y_test_cnn = le_cnn.transform(y_test)
y_train_cnn_categorical = to_categorical(y_train_cnn)
y_test_cnn_categorical = to_categorical(y_test_cnn)

print(f"CNN Train data shape: {X_train.shape}")
print(f"CNN Test data shape: {X_test.shape}")

# DEBUG: Print label encoder information
print("\n" + "="*40)
print("LABEL ENCODER DEBUG INFO")
print("="*40)
print(f"Unique labels found: {np.unique(y_train)}")
print(f"Label encoder classes: {le_cnn.classes_}")
print(f"Label mapping:")
for i, class_name in enumerate(le_cnn.classes_):
    print(f"  {class_name} -> {i}")

print(f"\nFirst 10 original labels: {y_train[:10]}")
print(f"First 10 encoded labels: {y_train_cnn[:10]}")

print("="*40)

# Building the CNN model
cnn_model = Sequential([
    # Use the EXACT same architecture as fire detection
    layers.Conv2D(8, (5, 5), activation='sigmoid', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(16, (3, 3), activation='sigmoid'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (5, 5), activation='sigmoid'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='sigmoid'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(2, activation='softmax')
])


# Compile the model with SGD optimizer
opt = SGD(learning_rate=0.01)
cnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print("\nCNN Model Architecture:")
cnn_model.summary()

# Data Augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Train the model with augmented data
H = cnn_model.fit(
    aug.flow(X_train, y_train_cnn_categorical, batch_size=64),
    validation_data=(X_test, y_test_cnn_categorical),
    epochs=25,
    verbose=1
)


# Evaluate CNN
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test_cnn_categorical, verbose=0)
print(f"\nCNN Test Loss: {cnn_loss:.4f}, Test Accuracy: {cnn_accuracy:.4f}")

# Save the CNN model
cnn_model.save("./models/cat_dog_classifier_cnn_sigmoid.h5")

# Plot training history
plt.figure(figsize=(12, 4))

plt.plot(H.history['loss'], label='train loss')
plt.plot(H.history['val_loss'], label='val loss')
plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='val accuracy')

plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('CNN Training and Validation Loss/Accuracy Metrics')
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "="*50)
print("CNN TRAINING COMPLETE!")
print("="*50)

