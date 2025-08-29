# Brain Tumor Detection with a Custom CNN using TensorFlow & Keras

## Project Overview
This project documents the end-to-end process of building, training, and evaluating a Convolutional Neural Network (CNN) for brain tumor detection. The model is a binary classifier designed to distinguish between MRI images that show a brain tumor ('yes') and those that do not ('no'). This repository showcases a complete machine learning workflow, including data preprocessing, data augmentation, model architecture design, iterative training, and performance analysis.

---

## Dataset
The project utilizes the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Content:** The dataset contains 253 MRI images across two classes.
  - `yes` (Tumor): 155 images
  - `no` (No Tumor): 98 images
- **Data Split:** The data was randomly partitioned into:
  - **70% Training Set:** 176 images
  - **15% Validation Set:** 37 images
  - **15% Test Set:** 37 images

---

## Methodology

### Data Preprocessing & Augmentation
Due to the limited dataset size, a key challenge was to prevent overfitting. This was addressed with a robust data augmentation strategy for the training set using Keras's `ImageDataGenerator`.
- **Normalization:** Pixel values were rescaled from the [0, 255] range to [0, 1].
- **Augmentation (Training Set Only):**
  - `zoom_range=0.2`
  - `shear_range=0.2`
  - `horizontal_flip=True`
- **Image Resizing:** All images were uniformly resized to `(224, 224)` pixels.

### Model Architecture
A custom Sequential CNN was built with the following layers:
- Conv2D (16 filters, 3x3 kernel, ReLU)
- Conv2D (36 filters, 3x3 kernel, ReLU) + MaxPool2D
- Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPool2D
- Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPool2D
- Dropout (rate=0.25) to regularize the feature extractor.
- Flatten
- Dense (64 units, ReLU)
- Dropout (rate=0.25) to regularize the classifier.
- **Output Layer:** Dense (1 unit, Sigmoid) for binary classification.

### Training & Optimization
The model was trained using an intelligent callback system to ensure efficiency and save the best-performing version:
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-entropy
- **Callbacks Used:**
  - `ModelCheckpoint`: Saved the model only when `val_accuracy` improved, ensuring the best model was preserved.
  - `EarlyStopping`: Monitored `val_accuracy` and halted training when the model stopped improving, preventing overfitting and saving time.
  - `ReduceLROnPlateau`: Monitored `val_loss` and automatically reduced the learning rate if the model's learning plateaued, allowing for finer convergence.

---

## Results & Analysis
The final model was selected based on its peak performance on the validation set during the most stable training run.

- **Peak Validation Accuracy:** **~86.5%**
- **Final Test Accuracy:** The best saved model achieved an accuracy of **78.4%** on the completely unseen test set.

![acc vs v-acc]
<img width="515" height="405" alt="graph" src="https://github.com/user-attachments/assets/b40e3ae0-3fdb-4798-ae39-fba9c89e66b8" />

The difference between the validation and test accuracy is a realistic outcome given the small dataset size, highlighting the impact of the specific random distribution of images in the validation and test splits. The test accuracy is the most honest measure of the model's generalization capability.

---

## How to Use
The notebook includes a section for testing the saved `best_model.h5` on a single image to make a real-time prediction.
1. Clone the repository.
2. Install the required libraries: `pip install -r requirements.txt`
3. Open the `Cancer_Detection_DL.ipynb` notebook and run the cells.

---

## Technologies Used
- TensorFlow & Keras
- NumPy
- Matplotlib
- Google Colab (GPU)
