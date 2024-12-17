import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Đường dẫn đến thư mục dữ liệu
train_dir = 'train'
validation_dir = 'valid'
test_dir = 'test'
target_size = (64, 64)
batch_size = 32
num_classes = 5

# Tiền xử lý dữ liệu
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=True
)
validation_generator = datagen.flow_from_directory(
    validation_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=True
)
test_generator = datagen.flow_from_directory(
    test_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Bước 1: Mô hình CNN
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Biên dịch mô hình CNN
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình CNN (dù chỉ 1 epoch để gọi mô hình)
history_cnn = model_cnn.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=1  # Huấn luyện 1 epoch để gọi mô hình
)

# Đánh giá mô hình CNN
cnn_test_loss, cnn_test_accuracy = model_cnn.evaluate(test_generator)
print(f"Độ chính xác trên tập test (CNN): {cnn_test_accuracy * 100:.2f}%")

# Bước 2: Trích xuất đặc trưng từ mô hình CNN
# Thực hiện một lần dự đoán để mô hình được gọi
model_cnn.predict(next(train_generator)[0])  # Thực hiện dự đoán cho một batch

# Kiểm tra số lớp của mô hình
print(f"Số lớp trong mô hình CNN: {len(model_cnn.layers)}")

# Trích xuất lớp 'Flatten', lớp này là lớp thứ 6 trong mô hình của bạn
feature_extractor = tf.keras.Model(inputs=model_cnn.input, outputs=model_cnn.get_layer('flatten').output)

def extract_features(generator):
    features = []
    labels = []
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = feature_extractor.predict(inputs_batch)
        features.append(features_batch)
        labels.append(labels_batch)
        if len(features) * batch_size >= generator.samples:
            break
    return np.vstack(features), np.argmax(np.vstack(labels), axis=1)

# Trích xuất đặc trưng từ tập train và test
train_features, train_labels = extract_features(train_generator)
test_features, test_labels = extract_features(test_generator)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear')
svm_model.fit(train_features, train_labels)

# Đánh giá mô hình SVM
svm_test_predictions = svm_model.predict(test_features)
svm_test_accuracy = accuracy_score(test_labels, svm_test_predictions)
print(f"Độ chính xác trên tập test (SVM): {svm_test_accuracy * 100:.2f}%")

# Báo cáo chi tiết cho SVM
print("Báo cáo chi tiết trên tập test (SVM):")
print(classification_report(test_labels, svm_test_predictions))

# Ma trận nhầm lẫn cho SVM
print("Ma trận nhầm lẫn (SVM):")
print(confusion_matrix(test_labels, svm_test_predictions))

# Vẽ biểu đồ so sánh
plt.figure(figsize=(14, 5))

# Đồ thị độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='CNN Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Validation Accuracy')
plt.axhline(y=svm_test_accuracy, color='r', linestyle='--', label='SVM Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("So sánh độ chính xác giữa CNN và SVM")

# Đồ thị hàm mất mát
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='CNN Train Loss')
plt.plot(history_cnn.history['val_loss'], label='CNN Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Hàm mất mát của CNN")

plt.tight_layout()
plt.show()
