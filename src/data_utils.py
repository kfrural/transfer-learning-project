import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=32,
        class_mode='categorical',
        target_size=(224, 224)
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=32,
        class_mode='categorical',
        target_size=(224, 224)
    )

    return train_generator, validation_generator
