import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, validation_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True)
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def plot_results(history):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(history.history['loss'])), history.history['loss'], label='Training Loss')
    plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], label='Training Accuracy')
    plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
