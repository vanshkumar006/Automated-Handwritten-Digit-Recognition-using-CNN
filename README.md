# Automated-Handwritten-Digit-Recognition-using-CNN
A robust CNN implementation for MNIST digit classification using TensorFlow Keras. Features BatchNormalization, Dropout, and Early Stopping. Includes comprehensive visualization tools for training history, confusion matrices, and prediction samples to analyze model performance.


1. Project Description
This project focuses on building a Convolutional Neural Network (CNN) capable of identifying handwritten digits with high precision. Using the MNIST dataset, which contains 60,000 training images and 10,000 testing images, the model mimics human visual processing to recognize numbers from 0 to 9.

The architecture is designed to be robust and efficient. It transforms raw pixel data (grayscale images) into probability scores for each digit. To ensure the model generalizes well to new, unseen handwriting, the project implements advanced deep learning techniques such as Batch Normalization (for stable training), Dropout (to prevent overfitting), and Early Stopping (to optimize training time).
Beyond simple classification, this project emphasizes interpretability. It includes a full visualization pipeline that generates training curves, confusion matrices, and side-by-side comparisons of actual vs. predicted labels, allowing for a deep analysis of where the model excels and where it confuses similar digits.



3. Libraries and Modules Used
The project relies on a powerful stack of Python libraries for data manipulation, deep learning, and visualization:

TensorFlow & Keras (tensorflow, keras, layers): Used to build the CNN architecture.

Modules: Conv2D (feature detection), MaxPooling2D (dimensionality reduction), BatchNormalization, Dropout, and Dense layers.

Callbacks: EarlyStopping was used to monitor validation loss and stop training automatically when the model stopped improving.

NumPy (numpy): Used for matrix operations, reshaping image arrays, and handling numerical data efficiently.

Matplotlib (matplotlib.pyplot):  Used to plot the training accuracy/loss graphs and visualize individual image predictions.

Seaborn (seaborn): Used to create the aesthetic and readable Heatmap for the Confusion Matrix.

Scikit-Learn (sklearn.metrics): Used to generate the detailed classification_report (Precision, Recall, F1-Score) and compute the confusion_matrix.



5. Project Results
The model demonstrated exceptional performance, converging quickly and achieving high accuracy on the test dataset.
Final Test Accuracy: 99.07%
Training Behavior: The model was set for 10 epochs but stopped early at Epoch 8 because it reached optimal performance, preventing wasted computational resources.
Loss Metrics: The final validation loss was extremely low (0.0356), indicating high confidence in predictions.

Class-Wise Performance (from Classification Report): The model showed consistent performance across all digits, with no significant bias toward any specific number.
Digit '1': Achieved a perfect 1.00 F1-Score.
Digit '5': Often the hardest to distinguish, yet the model achieved 0.99 precision.
Overall: The weighted average for Precision, Recall, and F1-Score was 0.99, confirming the model is highly reliable.



7. Conclusion
This Digit Recognizer project successfully demonstrates the power of Convolutional Neural Networks for computer vision tasks. By achieving a 99.07% accuracy rate, the model proves that a relatively lightweight architecture (approx. 420k parameters) can master the complexity of handwritten patterns when properly tuned with Batch Normalization and Dropout.
The visualization results confirm that the model effectively learned the unique spatial features of each digit (such as the loops in '8' or the slant of '7') rather than just memorizing the training data. This project serves as a robust baseline for more complex Optical Character Recognition (OCR) tasks.
