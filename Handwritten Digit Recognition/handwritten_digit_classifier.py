import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

X_train = train_data.iloc[:, 1:].values  # All columns except the first one
y_train = train_data.iloc[:, 0].values    # First column as labels

X_test = test_data.iloc[:, 1:].values    # All columns except the first one
y_test = test_data.iloc[:, 0].values      # First column as labels

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("="*60)
print("KNN CLASSIFIER")
print("="*60)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of KNN Classifier: {accuracy_knn * 100:.2f}%")

print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression: {accuracy * 100:.2f}%")


print("="*60)
print("SGD REGRESSION")
print("="*60)

sgd_model = SGDClassifier(max_iter=1000)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print(f"Accuracy of SGD Regression: {accuracy_sgd * 100:.2f}%")


print("="*60)
print("NEURAL NETWORK CLASSIFIER")
print("="*60)

# Build a simple neural network
nn_model = Sequential()
nn_model.add(Dense(128, activation='sigmoid', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(64, activation='sigmoid'))
nn_model.add(Dense(10, activation='softmax'))

# Compile the model
nn_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # sparse_categorical_crossentropy for integer labels

# Train
nn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate
test_loss, test_acc = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {test_acc * 100:.2f}%")