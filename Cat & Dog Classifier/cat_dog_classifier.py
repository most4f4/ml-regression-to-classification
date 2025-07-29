import cv2
import glob
import numpy as np
import joblib
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


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
    
    
    img = cv2.resize(img, (64, 64))     # Resize the image to 64x64 pixels
    img = img / 255.0                   # Normalize the image between 0 and 1
    img = img.flatten()                 # Flatten the image to a 1D array

    train_data.append(img)                    # Append the flattened image to the data list

    # address example: dataset/cat/cat.1.jpg
    label = address.split('\\')[-2]     # Get the label from the file path
    train_labels.append(label)                # Append the label to the labels list

    if i % 200 == 0:
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
    
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten()

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


print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# KNN MODEL
print("\n" + "="*50)
print("K-NEAREST NEIGHBORS")
print("="*50)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, 'cat_dog_classifier_knn.z')

# EVALUATE KNN MODEL
pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, pred_knn)
print(f"Accuracy (KNN): {accuracy_knn:.4f}")

# LOGISTIC REGRESSION MODEL
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)  # Use scaled data
joblib.dump(logistic_model, 'cat_dog_classifier_logistic.z')

# EVALUATE LOGISTIC REGRESSION MODEL
pred_logistic = logistic_model.predict(X_test_scaled)
accuracy_logistic = accuracy_score(y_test, pred_logistic)
print(f"Accuracy (Logistic Regression): {accuracy_logistic:.4f}")

# SGD MODEL
print("\n" + "="*50)
print("STOCHASTIC GRADIENT DESCENT")
print("="*50)
sgd_model = SGDClassifier(max_iter=1000)  # Stochastic Gradient Descent Classifier
sgd_model.fit(X_train_scaled, y_train)  # Use scaled data
joblib.dump(sgd_model, 'cat_dog_classifier_sgd.z')  # Save the trained model to a file

# EVALUATE SGD MODEL
pred_sgd = sgd_model.predict(X_test_scaled)
accuracy_sgd = accuracy_score(y_test, pred_sgd)
print(f"Accuracy (SGD): {accuracy_sgd:.4f}")

# NEURAL NETWORK MODEL
print("\n" + "="*50)
print("NEURAL NETWORK")
print("="*50)

# Encode string labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded) # Convert to categorical for NN
y_test_categorical = to_categorical(y_test_encoded)

# Define a simple neural network using sigmoid activations
model = Sequential()
model.add(Dense(128, activation='sigmoid', input_shape=(64*64*3,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))  # Output layer for 2 classes

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_categorical, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy_nn = model.evaluate(X_test_scaled, y_test_categorical)
print(f"Accuracy (Neural Network with Sigmoid): {accuracy_nn:.2f}")

# Save model
model.save("cat_dog_classifier_nn_sigmoid.h5")

# FINAL COMPARISON
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"KNN Accuracy:                {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
print(f"Logistic Regression Accuracy: {accuracy_logistic:.4f} ({accuracy_logistic*100:.2f}%)")
print(f"SGD Accuracy:                {accuracy_sgd:.4f} ({accuracy_sgd*100:.2f}%)")
print(f"Neural Network Accuracy:     {accuracy_nn:.4f} ({accuracy_nn*100:.2f}%)")

# Find best model
models = ['KNN', 'Logistic Regression', 'SGD', 'Neural Network']
accuracies = [accuracy_knn, accuracy_logistic, accuracy_sgd, accuracy_nn]
best_idx = np.argmax(accuracies)
print(f"\nBest Model: {models[best_idx]} with {accuracies[best_idx]*100:.2f}% accuracy")