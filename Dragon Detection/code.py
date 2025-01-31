import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Step 1: Data Preparation
def download_and_prepare_data(data_dir):
    # Assuming you have a function to download and extract images to data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        # Replace with actual download logic for dragon images
        print("Please download dragon images to", data_dir)

    # Create train and validation generators
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator


# Step 2: Build the CNN Model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Step 3: Train the Model
def train_model(model, train_generator, validation_generator, epochs=10):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    return history


# Step 4: Detection and Bounding Box Annotation
def detect_and_draw_boxes(model, test_image_path):
    # 加载测试图像
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Unable to read the image file at {test_image_path}.")
        return

    # 转换为灰度图并二值化（根据你的图像类型调整）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 获取外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        # **收缩框的逻辑**: 让框变得更小
        shrink_factor = 0.8  # 收缩比例，0.8 表示收缩到 80%
        new_w, new_h = int(w * shrink_factor), int(h * shrink_factor)
        new_x, new_y = x + (w - new_w) // 2, y + (h - new_h) // 2

        # 绘制收缩后的矩形框
        cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

    # 保存图像
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at: {output_path}")


# Main Execution
data_dir = "./dragon_images"  # Replace with your data directory
download_and_prepare_data(data_dir)
train_gen, val_gen = download_and_prepare_data(data_dir)

model = build_model()
history = train_model(model, train_gen, val_gen, epochs=10)

# Test the model on an image
test_image_path = "./test_image.jpg"  # Replace with your test image path
detect_and_draw_boxes(model, test_image_path)
