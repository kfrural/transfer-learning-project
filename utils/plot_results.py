import matplotlib.pyplot as plt

def plot_results(history):
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
