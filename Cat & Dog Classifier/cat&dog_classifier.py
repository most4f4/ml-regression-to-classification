import cv2
import glob
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


data = []  # Initialize an empty list to hold the data
labels = []  # Initialize an empty list to hold the labels

files = glob.glob('dataset/*/*/*')  
print(f"Found {len(files)} files")

for i, address in enumerate(files):
    img = cv2.imread(address)

    if img is None:
        print(f"Failed to load image: {address}")
        continue  # skip this image
    
    
    img = cv2.resize(img, (64, 64))     # Resize the image to 64x64 pixels
    img = img / 255.0                   # Normalize the image between 0 and 1
    img = img.flatten()                 # Flatten the image to a 1D array

    data.append(img)                    # Append the flattened image to the data list

    # address example: dataset/cat/cat.1.jpg
    label = address.split('\\')[-2]     # Get the label from the file path
    labels.append(label)                # Append the label to the labels list

    if i % 200 == 0:
        print(f"[INFO] Processed {i} images")

data = np.array(data)       # Convert the list to a NumPy array
print(data.shape)           # Print the shape of the data array

labels = np.array(labels)   # Convert labels to a NumPy array
print(labels.shape)         # Print the shape of the labels array

X = data    # Features are the flattened images
y = labels  # Labels are the corresponding categories

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL
knn = KNeighborsClassifier()  # K-Nearest Neighbors Classifier
knn.fit(X_train, y_train)  # Fit the model
joblib.dump(knn, 'cat_dog_classifier_knn.z')  # Save the trained model to a file

# EVALUATE
pred_knn = knn.predict(X_test)  # Make predictions on the test set
accuracy_knn = accuracy_score(y_test, pred_knn)  # Calculate accuracy
print(f"Accuracy (KNN): {accuracy_knn:.2f}")


# MODEL
logistic_model = LogisticRegression(max_iter=1000)  # Logistic Regression Classifier
logistic_model.fit(X_train, y_train)  # Fit the model
joblib.dump(logistic_model, 'cat_dog_classifier_logistic.z')  # Save the trained model to a file

# EVALUATE
pred_logistic = logistic_model.predict(X_test)  # Make predictions on the test set
accuracy_logistic = accuracy_score(y_test, pred_logistic)  # Calculate accuracy
print(f"Accuracy (Logistic Regression): {accuracy_logistic:.2f}")


# MODEL
sgd_model = SGDClassifier(max_iter=1000)  # Stochastic Gradient Descent Classifier
sgd_model.fit(X_train, y_train)  # Fit the model
joblib.dump(sgd_model, 'cat_dog_classifier_sgd.z')  # Save the trained model to a file

# EVALUATE
pred_sgd = sgd_model.predict(X_test)  # Make predictions on the test set
accuracy_sgd = accuracy_score(y_test, pred_sgd)  # Calculate accuracy
print(f"Accuracy (SGD): {accuracy_sgd:.2f}")


# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)        # 'cat' -> 0, 'dog' -> 1
y_categorical = to_categorical(y_encoded)

# Train/test split with encoded labels
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Define a simple neural network using sigmoid activations
model = Sequential()
model.add(Dense(128, activation='sigmoid', input_shape=(64*64*3,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))  # Output layer for 2 classes

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_nn, y_train_nn, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy_nn = model.evaluate(X_test_nn, y_test_nn)
print(f"Accuracy (Neural Network with Sigmoid): {accuracy_nn:.2f}")

# Save model
model.save("cat_dog_classifier_nn_sigmoid.h5")