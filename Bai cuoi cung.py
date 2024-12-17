import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Đường dẫn đến thư mục dữ liệu
train_dir = 'train'
test_dir = 'test'


# 1. Load dữ liệu từ thư mục
def load_data(directory, target_size=(28, 28)):
    generator = ImageDataGenerator(rescale=1.0 / 255)
    data = generator.flow_from_directory(
        directory,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=1,
        class_mode='sparse',
        shuffle=False
    )
    images = []
    labels = []
    for _ in range(len(data)):
        img, label = next(data)  # Sử dụng `next(data)` để lấy ảnh và nhãn
        images.append(img[0, :, :, 0])  # Lấy ảnh grayscale
        labels.append(label[0])
    return np.array(images), np.array(labels)


# Load dữ liệu
print("Loading training data...")
x_train, y_train = load_data(train_dir)
print("Loading testing data...")
x_test, y_test = load_data(test_dir)


# 2. HOG + SVM
def extract_hog_features(images):
    features = []
    for img in images:
        img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về 64x64 cho HOG
        hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        features.append(hog_features)
    return np.array(features)


# 3. Huấn luyện và đánh giá mô hình
def train_and_evaluate_svm(features_train, labels_train, features_test, labels_test, model_name):
    print(f"Training {model_name}...")
    svm_model = svm.SVC(kernel='linear', C=1.0)
    svm_model.fit(features_train, labels_train)

    print(f"Predicting with {model_name}...")
    preds = svm_model.predict(features_test)

    # Đánh giá hiệu suất
    accuracy = accuracy_score(labels_test, preds)
    precision = precision_score(labels_test, preds, average='weighted')
    recall = recall_score(labels_test, preds, average='weighted')
    f1 = f1_score(labels_test, preds, average='weighted')

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
    return accuracy, precision, recall, f1


# 4. So sánh hai mô hình
print("Extracting HOG features...")
hog_train = extract_hog_features(x_train)
hog_test = extract_hog_features(x_test)

# Sử dụng đặc trưng thô (raw features)
print("Flattening raw image features...")
raw_train = x_train.reshape(x_train.shape[0], -1)
raw_test = x_test.reshape(x_test.shape[0], -1)

# Huấn luyện và đánh giá
hog_results = train_and_evaluate_svm(hog_train, y_train, hog_test, y_test, "SVM + HOG")
raw_results = train_and_evaluate_svm(raw_train, y_train, raw_test, y_test, "SVM + Raw Features")

# So sánh kết quả
print("Comparison of Results:")
print(
    f"SVM + HOG: Accuracy={hog_results[0]:.4f}, Precision={hog_results[1]:.4f}, Recall={hog_results[2]:.4f}, F1-Score={hog_results[3]:.4f}")
print(
    f"SVM + Raw Features: Accuracy={raw_results[0]:.4f}, Precision={raw_results[1]:.4f}, Recall={raw_results[2]:.4f}, F1-Score={raw_results[3]:.4f}")
