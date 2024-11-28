import os
import sys
from utils.download_data import download_and_extract_data
from data_utils import load_data
from model import create_model
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_model(train_generator, validation_generator):
    input_shape = (224, 224, 3)
    num_classes = len(train_generator.class_indices) + 1  

    model = create_model(input_shape, num_classes)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    return history

def plot_results(history):
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dataset_name = 'cats-and-dogs'
    target_dir = 'data/'
    download_and_extract_data(dataset_name, target_dir)

    train_dir = 'data/training_set'
    validation_dir = 'data/test_set'

    train_generator, validation_generator = load_data(train_dir, validation_dir)

    history = train_model(train_generator, validation_generator)

    plot_results(history)
