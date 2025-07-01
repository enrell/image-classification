# TensorFlow and tf.keras, more info at: https://www.tensorflow.org/guide/keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for better display
import matplotlib

matplotlib.use("Agg")  # Use Agg backend which works well in most environments
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode to avoid warnings

# debug: prints TensorFlow's version
print(tf.__version__)

# imports the dataset (it actually downloads it)
fashion_mnist = keras.datasets.fashion_mnist

# sets up the train and test sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# debug: prints info about the shape and the size of the sets
print("Shape of the training images", train_images.shape)
print("Number of training labels: ", len(train_labels))
print("Shape of the test images", test_images.shape)
print("Number of test labels: ", len(test_labels))

# displays the first 25 imagens of the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("fashion_mnist_samples_raw.png", dpi=150, bbox_inches="tight")
print("Saved plot: fashion_mnist_samples_raw.png")
plt.show()

# values between 0 to 255 are changed to 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("Shape of the training images:", train_images.shape)
print("Number of training labels: ", len(train_labels))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap="gray")
    plt.xlabel(class_names[train_labels[i]])
plt.suptitle("Sample Images from Fashion MNIST Dataset", fontsize=16)
plt.savefig("fashion_mnist_samples_normalized.png", dpi=150, bbox_inches="tight")
print("Saved plot: fashion_mnist_samples_normalized.png")
plt.show()


# neural network and training parameters - OPTIMIZED FOR HIGHER ACCURACY
# see https://keras.io/api/layers/ for more details
hidden_layer_size = 256
activation_function = "leaky_relu"
leaky_relu_negative_slope = 0.1
kernel_initializer = "he_normal"
training_optimizer = "adam"
learning_rate = 0.001
number_of_epochs = 150
dropout_rate = 0.3
batch_size = 64


# creating the ANN
model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28), name="input_layer"),
        keras.layers.Flatten(),
        keras.layers.Dense(
            hidden_layer_size,
            kernel_initializer=kernel_initializer,
            name="hidden_layer",
        ),
        keras.layers.LeakyReLU(
            negative_slope=leaky_relu_negative_slope, name="leaky_relu_activation"
        ),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, name="output_layer"),
    ]
)

model.summary()

# compiling the model with optimized settings
model.compile(
    optimizer=training_optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Training with learning rate scheduling and early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=30,
    restore_best_weights=True,
    verbose=1,
)

# Add learning rate reduction on plateau
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_images,
    train_labels,
    epochs=number_of_epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping, lr_scheduler],
)

# Create training history plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
ax1.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="red")
ax1.set_title("Model Accuracy Over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history["loss"], label="Training Loss", color="blue")
ax2.plot(history.history["val_loss"], label="Validation Loss", color="red")
ax2.set_title("Model Loss Over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
print("Saved plot: training_history.png")
plt.show()

print("\n--- Analyzing Hidden Layer Activations ---")

# Extract the hidden layer output using the model's functional API
hidden_layer_output = model.get_layer(name="leaky_relu_activation").output
model_input = model.layers[0].input

# Get activations for a subset of the training data for efficiency
activation_model = keras.Model(inputs=model_input, outputs=hidden_layer_output)

activations = activation_model.predict(train_images[:1000])

# Calculate the percentage of zero-valued activations for relu usage
percent_zeros = np.mean(activations == 0) * 100
percent_negative = np.mean(activations < 0) * 100
print(
    f"{percent_zeros:.2f}% of hidden layer activations are exactly zero (for this sample)."
)
print(
    f"{percent_negative:.2f}% of hidden layer activations are negative (showing LeakyReLU working)."
)

# Plot histogram of activations
flat_activations = activations.flatten()
plt.figure(figsize=(10, 5))
plt.hist(flat_activations, bins=50, color="blue", alpha=0.7)
plt.title(
    f"Distribution of Activations (LEAKY RELU, slope={leaky_relu_negative_slope})"
)
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.xlim(-2, 10)
plt.grid(True)
plt.tight_layout()
plt.savefig("activation_distribution.png", dpi=150, bbox_inches="tight")
print("Saved plot: activation_distribution.png")
plt.show()

print("\nExecution ended.")

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest accuracy:", test_acc)

print("execution ended")

print("\n--- Checking for Dying Neurons ---")

all_train_activations = activation_model.predict(train_images, batch_size=256)

# For LeakyReLU, check neurons that have very low maximum activation (close to zero)
max_activations_per_neuron = np.max(all_train_activations, axis=0)
min_activations_per_neuron = np.min(all_train_activations, axis=0)

dying_threshold = (
    1e-4  # More lenient threshold for LeakyReLU with higher negative slope
)
dying_neurons = np.sum(max_activations_per_neuron < dying_threshold)
total_neurons = hidden_layer_size

print(
    f"Neurons with max activation < {dying_threshold}: {dying_neurons} out of {total_neurons}"
)
print(f"Percentage of dying neurons: {(dying_neurons / total_neurons) * 100:.2f}%")

# Additional analysis for LeakyReLU
mean_activations = np.mean(all_train_activations, axis=0)
std_activations = np.std(all_train_activations, axis=0)

print(f"\nActivation Statistics:")
print(
    f"Mean activation range: [{np.min(mean_activations):.4f}, {np.max(mean_activations):.4f}]"
)
print(
    f"Std activation range: [{np.min(std_activations):.4f}, {np.max(std_activations):.4f}]"
)
print(f"Neurons with negative mean activation: {np.sum(mean_activations < 0)}")

if dying_neurons > 0:
    print(f"\nWarning: {dying_neurons} neurons might be dying (very low activation).")
else:
    print("\nGood: No dying neurons detected with LeakyReLU!")
    print("LeakyReLU successfully prevents the dying neuron problem.")

    # Get the model predictions for the test set
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy and simple confidence interval for each class
    class_accuracies = {}
    for i in range(len(class_names)):
        class_indices = np.where(test_labels == i)[0]
        total_samples = len(class_indices)
        if total_samples > 0:
            correct_predictions = np.sum(
                predicted_labels[class_indices] == test_labels[class_indices]
            )
            accuracy = correct_predictions / total_samples
            class_accuracies[class_names[i]] = accuracy

            # Calculate standard error for confidence interval
            std_error = np.sqrt(accuracy * (1 - accuracy) / total_samples)
            print(f"{class_names[i]}: {accuracy:.3f} ± {1.96 * std_error:.3f}")
        else:
            class_accuracies[class_names[i]] = 0
            print(f"{class_names[i]}: No samples")

    # Enhanced bar plot of accuracies
    labels = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    print("\n--- Per-Class Accuracy Analysis ---")

    plt.figure(figsize=(15, 8))
    bars = plt.bar(
        labels, accuracies, color="skyblue", alpha=0.7, edgecolor="navy", linewidth=1.2
    )
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        "Per-Category Accuracy on Test Set (Fashion MNIST with LeakyReLU)", fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add overall accuracy line
    overall_acc = float(np.mean(accuracies))
    plt.axhline(
        y=overall_acc,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Overall Accuracy: {overall_acc:.3f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("per_category_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved plot: per_category_accuracy.png")
    plt.show()

    # Summary statistics
    print(f"\n--- Summary Statistics ---")
    print(
        f"Best performing category: {labels[np.argmax(accuracies)]} ({max(accuracies):.3f})"
    )
    print(
        f"Worst performing category: {labels[np.argmin(accuracies)]} ({min(accuracies):.3f})"
    )
    print(f"Standard deviation of accuracies: {np.std(accuracies):.3f}")
    print(f"Range of accuracies: {max(accuracies) - min(accuracies):.3f}")
