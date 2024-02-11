# Plant Disease Recognition

## Overview

This project aims to develop a model for classifying images of plants and identifying whether they are diseased or healthy. The recognition model utilizes deep learning techniques and is based on the TensorFlow and Keras frameworks.

## Why This Project is Useful

Plant disease recognition is crucial for early detection and intervention, enabling farmers and agriculturalists to take timely actions to prevent the spread of diseases. By automating the identification process through computer vision, this project contributes to more efficient and effective plant health monitoring.

## Features

- Image classification for healthy and diseased plants.
- Data augmentation for improved model generalization.
- Transfer learning using a pre-trained VGG16 model.
- Easy integration with different datasets.

## Setup

### Requirements

- Python 3.x
- TensorFlow
- Keras
- Other dependencies (specified in `requirements below`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/KannurAnudeepRaj/plant-disease-recognition.git
   ```

2. Change to the project directory:

   ```bash
   cd plant-disease-recognition
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

1. Prepare your dataset with labeled images of healthy and diseased plants.

2. Organize your dataset into the following directory structure:

   ```plaintext
   ├── dataset
   │   ├── train
   │   │   ├── healthy
   │   │   └── diseased
   │   ├── validation
   │   │   ├── healthy
   │   │   └── diseased
   │   └── test
   │       ├── healthy
   │       └── diseased
   ```

   Ensure that the number of classes matches the specified classes in the `class_mode` parameter of the data generators in the script.

### Training

1. Open `plant_disease_recognition.py` in a code editor.

2. Update the `train_dir`, `validation_dir`, and `test_dir` variables with the correct paths to your dataset.

3. Adjust other hyperparameters and model configurations if needed.

4. Run the script:

   ```bash
   python plant_disease_recognition.py
   ```

### Model Evaluation

1. After training, the model will be saved as `plant_disease_recognition_model.h5`.

2. Evaluate the model on your test set:

   ```bash
   python evaluate_model.py
   ```

## License

This project is licensed under the [MIT License](LICENSE).



## Requirements
```requirment.txt
tensorflow==2.7.0
keras==2.6.0
numpy==1.21.2
matplotlib==3.4.3
scikit-learn==0.24.2
```

This list includes TensorFlow and Keras for deep learning, NumPy for numerical operations, Matplotlib for plotting, and scikit-learn for additional machine learning utilities.

Remember to adjust the versions based on your preferences or the compatibility of the libraries you are using. You can install these dependencies by running:

```bash
pip install -r requirements.txt
```
