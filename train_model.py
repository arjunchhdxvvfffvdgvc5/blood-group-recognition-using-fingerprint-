import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from app import FingerprintProcessor, BloodGroupClassifier

def train_model():
    # Path to your training data
    data_path = 'training_data'
    blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    features = []
    labels = []
    
    # Process each blood group folder
    for blood_group in blood_groups:
        group_path = os.path.join(data_path, blood_group)
        if not os.path.exists(group_path):
            continue
            
        print(f"Processing {blood_group} images...")
        
        # Process all images in the blood group folder
        for img_path in glob(os.path.join(group_path, '*.*')):
            try:
                # Read and process image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Could not read image: {img_path}")
                    continue
                    
                # Process fingerprint
                processor = FingerprintProcessor()
                enhanced = processor.enhance_fingerprint(image)
                image_features = processor.extract_features(enhanced)
                
                features.append(image_features)
                labels.append(blood_group)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    if not features:
        print("No training data found!")
        return
        
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training model...")
    classifier = BloodGroupClassifier()
    classifier.train_model(X_train, y_train)
    
    # Test accuracy
    accuracy = classifier.model.score(X_test, y_test)
    print(f"Training completed! Test accuracy: {accuracy * 100:.2f}%")
    print(f"Total samples: {len(y)}")

if __name__ == "__main__":
    train_model() 