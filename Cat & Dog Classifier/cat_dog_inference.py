import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("./models/cat_dog_classifier_nn_sigmoid.h5")

image = cv2.imread('./test_images/cat_test2.jpg')  # Load a test image
image = cv2.resize(image, (64, 64))  # Resize the image to 64x64 pixels
image = image / 255.0  # Normalize the image
image = image.flatten()  # Flatten the image to a 1D array

# Make a prediction
prediction = model.predict(np.array([image]))

predicted_label = np.argmax(prediction, axis=1)[0]

label_map = {0: "Cat", 1: "Dog"}
print(f"Prediction: {label_map[predicted_label]}")    

# Print the predicted label
print(f"Predicted label: {predicted_label[0]}")  # Print the predicted label
cv2.imshow('Test Image', image.reshape(64, 64, 3))

cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window

