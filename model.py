import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')  # Set TkAgg backend
import matplotlib.pyplot as plt


# Load satellite images (considering RGB images for simplicity)
def load_images(file_paths, target_size=(256, 256)):
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)  # Resize the image to a consistent size
        images.append(img)
    return np.array(images)

# Perform color-based segmentation
def segment_image(img):
    # Color segmentation parameters (modify as needed)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([100, 100, 100])
    mask = cv2.inRange(img, lower_bound, upper_bound)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    return segmented_img

# Extract features from images (simple color histogram for demonstration)
def extract_features(images):
    features = []
    for img in images:
        # Perform segmentation
        segmented_img = segment_image(img)
        
        # Flatten the segmented image for simplicity
        flattened_segment = segmented_img.flatten()
        
        # Concatenate the flattened segment with color histogram features
        hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        features.append(np.concatenate((flattened_segment, hist.flatten())))
        
    return np.array(features)

# Create labels (0 for non-deforested, 1 for deforested)
def create_labels(num_samples, label_deforested_ratio=0.1):
    labels = np.zeros(num_samples)
    num_deforested = int(num_samples * label_deforested_ratio)
    labels[:num_deforested] = 1
    np.random.shuffle(labels)
    return labels

# Load and preprocess satellite images
file_paths = ["./img/image1.jpg", "./img/image2.jpeg", "./img/image3.jpg", "./img/image4.jpg"]
images = load_images(file_paths)
features = extract_features(images)

# Create labels
num_samples = len(file_paths)
labels = create_labels(num_samples)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a simple Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Segmentation and visualization
for i, img in enumerate(images):
    segmented_img = segment_image(img)
    
    # Visualize the original and segmented images
    plt.subplot(2, len(images), i + 1)
    plt.imshow(img)
    plt.title('Original')
    
    plt.subplot(2, len(images), i + len(images) + 1)
    plt.imshow(segmented_img)
    plt.title('Segmented')

plt.show()
