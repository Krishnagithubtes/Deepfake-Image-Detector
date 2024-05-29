import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.metrics import f1_score
from PIL import Image


def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.cast(tf.equal(parts[-2], 'Real'), tf.float32)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img /= 255.0
    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def create_test_dataset(num_images=None):
    data_dir = os.path.join(TEST_DIR)
    list_ds = tf.data.Dataset.list_files(str(data_dir + '/*/*'), shuffle=True)

    if num_images:
        list_ds = list_ds.take(num_images)

    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = labeled_ds.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 8), constrained_layout=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.show()


def display_sample_predictions(model, dataset, num_samples=10):
    class_names = ['Fake', 'Real']
    plt.figure(figsize=(20, 10))
    sample_count = 0

    for images, labels in dataset.unbatch().shuffle(buffer_size=1000).take(num_samples):
        predictions = model.predict(tf.expand_dims(images, 0))
        predicted_label = 'Real' if predictions[0][0] > 0.5 else 'Fake'
        true_label = 'Real' if labels.numpy() == 1 else 'Fake'

        is_correct = predicted_label == true_label
        title_color = 'green' if is_correct else 'red'

        sample_count += 1
        plt.subplot(2, 5, sample_count)
        plt.imshow(images.numpy().astype("float32"))
        plt.title(f"Predicted: {predicted_label}\nTrue: {true_label}", color=title_color, fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def get_test_files():
    data_dir = os.path.join(TEST_DIR)
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'Fake')

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    return real_files, fake_files


def test_model():
    if MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")

    model = load_model(MODEL_SAVE_PATH)
    print("Model loaded successfully.")

    real_files, fake_files = get_test_files()
    print(f"Number of Real images: {len(real_files)}")
    print(f"Number of Fake images: {len(fake_files)}")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("Error: Test dataset is missing one of the classes.")
        return

    y_true = []
    y_probs = []

    print("Making predictions on the test dataset...")
    for class_type, file_list in [('Real', real_files), ('Fake', fake_files)]:
        for file_path in file_list:
            img = Image.open(file_path).convert('RGB')
            img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)

            prediction = model.predict(img_batch)
            y_probs.append(prediction[0][0])
            y_true.append(1 if class_type == 'Real' else 0)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    best_threshold, best_f1 = find_best_threshold(y_true, y_probs)
    print(f"Best threshold: {best_threshold:.2f}, Best F1-score: {best_f1:.2f}")

    y_pred = (y_probs > best_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'], title='Test Confusion Matrix')

    print('Test Classification Report')
    target_names = ['Fake', 'Real']
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Displaying sample predictions:")
    sample_test_dataset = create_test_dataset(num_images=10)
    display_sample_predictions(model, sample_test_dataset, num_samples=10)


if __name__ == '__main__':
    test_model()
