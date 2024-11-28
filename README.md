# Transfer Learning Project

This project demonstrates how to apply **Transfer Learning** for image classification using a pre-trained model, specifically **MobileNetV2**, to classify image categories in a provided dataset.

## Project Structure

The project structure is organized as follows:

- `data/`: Contains training and validation data.
    - `train/`: Directory with training images, divided by category.
    - `validation/`: Directory with validation images, divided by category.
- `src/`: Contains the main code scripts.
    - `model.py`: Definition of the Transfer Learning model.
    - `data_utils.py`: Functions for loading and processing data.
    - `main.py`: Main script for training the model and visualizing results.
- `utils/`: Helper functions for data download and result plotting.
    - `download_data.py`: Script to download and extract the dataset.
    - `plot_results.py`: Functions to visualize performance graphs during training.
- `requirements.txt`: List of dependencies required to run the project.
- `.gitignore`: Files and directories that will not be versioned, such as data and caches.
- `README.md`: Project documentation.

## Prerequisites

Make sure you have the following prerequisites installed on your system:

- Python 3.x
- Pip (for managing packages)

## Installation

1. **Clone this repository:**

```bash
git clone https://github.com/your-username/transfer-learning-project.git
cd transfer-learning-project
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. **Install the project dependencies:**

```bash
pip install -r requirements.txt
```

## Download and Prepare the Data

1. **Download and extract the data:**

The `download_data.py` script can be used to download and extract the data directly from a provided URL.

- Modify the `url` variable in the `download_data.py` file to the URL of your dataset.
- Then, run the script to download the data:

```bash
python utils/download_data.py
```

2. **Organize the data:**

Make sure the images are organized into the expected directory structure, with training (`train/`) and validation (`validation/`) directories.

## Model Training

1. **Run the main script to train the model:**

```bash
python src/main.py
```

This will load the training and validation data, train the model using Transfer Learning with MobileNetV2, and display performance graphs (loss and accuracy) during the training process.

## Result Visualization

During training, graphs will be automatically generated to show the evolution of **Loss** and **Accuracy** for both the training and validation sets.

## Model Architecture

The model uses **MobileNetV2** as a base, removing the final layer (top) and adding new layers to adapt to your specific problem. The model is compiled using the **Adam** optimizer and the **categorical_crossentropy** loss function.

## Contributions

Feel free to contribute to the project! You can:
- Open issues to suggest improvements or report bugs.
- Fork the repository and submit a pull request with your contributions.