# ===== IMPORTS =====
import zipfile
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report

# ===== DATA EXTRACTION =====
zip_path = '/content/drive/My Drive/archive.zip'
extract_to = '/content/brain_tumor'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extraction Completed!")
print("Extracted folders:")
print(os.listdir(extract_to))

# ===== DATA EXPLORATION =====
train_dir = '/content/brain_tumor/Training'
test_dir = '/content/brain_tumor/Testing'

print("Train Folders:", os.listdir(train_dir))
print("Test Folders:", os.listdir(test_dir))

# Base directory
base_dir = '/content/brain_tumor'
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Testing')

print("Training directory:", train_dir)
print("Testing directory:", test_dir)
print("\nSubfolders inside Training:")
print(os.listdir(train_dir))
print("\nSubfolders inside Testing:")
print(os.listdir(test_dir))

# ===== VISUALIZE SAMPLE IMAGES =====
train_dir = '/content/brain_tumor/Training'
tumor_types = os.listdir(train_dir)

plt.figure(figsize=(12, 8))

for i, tumor in enumerate(tumor_types):
    tumor_path = os.path.join(train_dir, tumor)
    sample_imgs = random.sample(os.listdir(tumor_path), 2)
    
    for j, img_name in enumerate(sample_imgs):
        img_path = os.path.join(tumor_path, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(len(tumor_types), 2, i*2 + j + 1)
        plt.imshow(img)
        plt.title(tumor)
        plt.axis('off')

plt.tight_layout()
plt.show()

# ===== CLASS DISTRIBUTION =====
for directory in [train_dir, test_dir]:
    print(f"\n📁Folder: {directory}")
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name:12} : {len(os.listdir(class_path))} images")

# ===== CREATE VALIDATION SET =====
base_dir = '/content/brain_tumor'
val_dir = os.path.join(base_dir, 'Validation')
os.makedirs(val_dir, exist_ok=True)

classes = os.listdir(train_dir)

for cls in classes:
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    src = os.path.join(train_dir, cls)
    dst = os.path.join(val_dir, cls)
    files = os.listdir(src)
    random.shuffle(files)
    
    val_count = int(len(files) * 0.2)
    val_files = files[:val_count]
    
    for f in val_files:
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

print("✅ Validation data created successfully!")

# ===== DATA GENERATORS =====
train_dir = "/content/brain_tumor/Training"
test_dir = "/content/brain_tumor/Testing"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ===== CNN MODEL =====
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===== CALLBACKS =====
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

lr_scheduler = LearningRateScheduler(scheduler)
callbacks = [early_stop, checkpoint, lr_scheduler]

# ===== TRAINING =====
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ===== TEST EVALUATION =====
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"\n Test Accuracy: {test_acc:.2f}")

# ===== CONFUSION MATRIX & CLASSIFICATION REPORT =====
class_names = list(train_generator.class_indices.keys())

test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===== PLOTTING TRAINING HISTORY =====
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ===== SAVE FINAL MODEL =====
model.save('brain_tumor_classifier_final.h5')
print("✅ Model saved as 'brain_tumor_classifier_final.h5'")
