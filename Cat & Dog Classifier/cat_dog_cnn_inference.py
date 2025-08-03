
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.models import load_model
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the saved CNN model
loaded_cnn_model = load_model("./models/cat_dog_classifier_cnn_sigmoid.h5")
print("CNN model loaded successfully!")

# Recreate the label encoder 
le_cnn = LabelEncoder()
le_cnn.fit(['Cat', 'Dog']) 
print("Label encoder created")

# Test all images in test_images folder
print("\n" + "="*50)
print("TESTING ALL IMAGES IN test_images FOLDER")
print("="*50)

all_test_files = glob.glob('test_images/*')
print(f"Found {len(all_test_files)} images in test_images folder")

results = []

for image_path in all_test_files:
    # Load and preprocess image
    test_img = cv2.imread(image_path)
    
    if test_img is None:
        continue
    
    # Store original for display
    original_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Preprocess exactly like training
    test_img = cv2.resize(test_img, (32, 32))
    test_img = test_img / 255.0
    test_img = np.expand_dims(test_img, axis=0)
    
    # Predict
    prediction = loaded_cnn_model.predict(test_img, verbose=0)

    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    # np.argmax returns the index of the maximum value along the specified axis
    # Here, axis=1 means we are looking for the maximum value across the classes for each sample in the batch
    # [0] is used to get the index for the first (and only) sample in the batch
    
    confidence = np.max(prediction) * 100
    predicted_class = le_cnn.classes_[predicted_class_idx]
    # le_cnn.classes_ returns the original class labels in the order they were encoded
    # predicted_class_idx gives the index of the predicted class
    
    # Display image
    plt.figure(figsize=(6, 4))
    plt.imshow(original_img)
    
    # Color based on confidence
    if confidence > 70:
        color = 'green'
    elif confidence > 50:
        color = 'orange'
    else:
        color = 'red'
        
    plt.title(f'{predicted_class.upper()} ({confidence:.1f}%)', 
             fontsize=14, fontweight='bold', color=color)
    plt.axis('off')
    plt.show()
    
    results.append({
        'image': image_path,
        'prediction': predicted_class,
        'confidence': confidence
    })
    
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class.upper()} ({confidence:.1f}%)")
    print("-" * 30)

# Print summary table
print("\n" + "="*60)
print("SUMMARY OF ALL PREDICTIONS")
print("="*60)
print(f"{'Image':<30} {'Prediction':<12} {'Confidence':<12}")
print("-" * 60)

for result in results:
    img_name = result['image'].split('/')[-1].split('\\')[-1]
    prediction = result['prediction'].upper()
    confidence = f"{result['confidence']:.1f}%"
    
    print(f"{img_name:<30} {prediction:<12} {confidence:<12}")

# Statistics
if results:
    cat_count = sum(1 for r in results if r['prediction'] == 'Cat')
    dog_count = sum(1 for r in results if r['prediction'] == 'Dog')
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print("\n" + "="*40)
    print("STATISTICS")
    print("="*40)
    print(f"Total images tested: {len(results)}")
    print(f"Cat predictions: {cat_count}")
    print(f"Dog predictions: {dog_count}")
    print(f"Average confidence: {avg_confidence:.1f}%")
    
    high_confidence = sum(1 for r in results if r['confidence'] > 70)
    print(f"High confidence predictions (>70%): {high_confidence}")
else:
    print("❌ No images found to test!")
    print("💡 Make sure you have images in the 'test_images' folder")

print("\n" + "="*60)
print("CNN EVALUATION COMPLETE!")
print("="*60)

