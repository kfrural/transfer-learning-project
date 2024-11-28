import tensorflow as tf
from data_utils import load_data
from model import create_model

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

if __name__ == "__main__":
    from utils.download_data import download_and_extract_data
    download_and_extract_data()

    train_dir = 'data/train'
    validation_dir = 'data/validation'
    train_generator, validation_generator = load_data(train_dir, validation_dir)

    history = train_model(train_generator, validation_generator)

    plot_results(history)
