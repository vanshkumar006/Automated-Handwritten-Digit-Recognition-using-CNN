import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Make the plots look nice
sns.set_theme(style='whitegrid')

# --- CONFIGURATION ---
# Easy to tweak parameters at the top
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SHAPE = (28, 28, 1) # Height, Width, Channels (Grayscale)

def load_data():
    """
    Loads MNIST data and normalizes it.
    """
    print("Step 1: Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    # This helps the neural network converge faster
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNNs expect 4 dimensions: (Batch, Height, Width, Channels)
    # We need to add the 'Channels' dimension (1 for grayscale)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"   -> Training samples: {x_train.shape[0]}")
    print(f"   -> Testing samples:  {x_test.shape[0]}")
    
    return x_train, y_train, x_test, y_test

def build_model():
    """
    Creates a Convolutional Neural Network (CNN).
    Structure: Conv2D -> BatchNormalization -> ReLU -> Pooling
    """
    print("Step 2: Building the model architecture...")
    
    inputs = keras.Input(shape=IMG_SHAPE)

    # First Convolutional Block
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second Convolutional Block (captures more complex features)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and Output
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x) # Prevents overfitting
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x) # 10 digits (0-9)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Digit_Recognizer_CNN")
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )
    
    return model

def plot_history(history):
    """
    Plots accuracy and loss over epochs.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    
    plt.show()

def visualize_predictions(model, x_test, y_test):
    """
    Shows a few random images with their predicted labels.
    """
    print("Step 5: Visualizing predictions...")
    predictions = model.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(12, 6))
    indices = np.random.choice(len(x_test), 10, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        
        # Green title if correct, Red if wrong
        true_label = y_test[idx]
        pred_label = pred_labels[idx]
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    return pred_labels

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    
    # 1. Get Data
    x_train, y_train, x_test, y_test = load_data()
    
    # 2. Build Model
    model = build_model()
    model.summary()
    
    # 3. Train
    print("Step 3: Training (this might take a minute)...")
    # Early stopping ensures we don't waste time if the model stops improving
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    # 4. Results
    print("Step 4: Evaluating performance...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    
    plot_history(history)
    
    # 5. Deep Dive Analysis
    pred_labels = visualize_predictions(model, x_test, y_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred_labels))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()