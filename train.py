import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data_loader import get_datasets
from model import build_model
from config import *
import os
import datetime
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.utils import class_weight
import math

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)

def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(int)
        f1 = classification_report(y_true, y_pred, output_dict=True)['binary']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('training_accuracy.png')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    plt.show()

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow Version: ", tf.__version__)

    if MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")

    train_dataset, validation_dataset = get_datasets()
    model = build_model()

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    log_dir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    y_train = []
    for _, labels in train_dataset:
        y_train.extend(labels.numpy())
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    print(f"Class weights: {class_weights}")

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        callbacks=[checkpoint, early_stop, tensorboard_callback, lr_callback],
        class_weight=class_weights
    )

    plot_training(history)

    validation_dataset = validation_dataset.unbatch().batch(BATCH_SIZE)
    y_true = []
    y_pred = []

    for images, labels in validation_dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.where(predictions > 0.5, 1, 0).flatten())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake', 'Real'], rotation=45)
    plt.yticks(tick_marks, ['Fake', 'Real'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

    print('Classification Report')
    target_names = ['Fake', 'Real']
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == '__main__':
    main()
