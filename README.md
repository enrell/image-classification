# Fashion MNIST Neural Network Analysis

This project implements an neural network for Fashion MNIST classification using TensorFlow/Keras. The model is designed to achieve ~90% accuracy by addressing common issues like the dying ReLU problem.

## 🎯 Key Features

- **LeakyReLU activation** to prevent dying neurons
- **Larger hidden layer** (256 neurons)
- **Advanced training techniques** with learning rate scheduling
- **Comprehensive analysis** of model performance and neuron health
- **Detailed visualizations** of training progress and results

## 📊 Dataset

**Fashion-MNIST** is a dataset of Zalando's article images consisting of:
- **Training set**: 60,000 examples
- **Test set**: 10,000 examples  
- **Image size**: 28x28 grayscale images
- **Classes**: 10 fashion categories

### Classes
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

**Dataset Source**: [Kaggle Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## 🏗️ Model Architecture

### Sequential Neural Network
```
Input Layer (28x28) 
    ↓
Flatten Layer
    ↓
Dense Hidden Layer (256 neurons)
    ↓
LeakyReLU Activation (slope=0.1)
    ↓
Dropout (0.3)
    ↓
Output Layer (10 neurons)
```

### Key Parameters
- **Hidden layer size**: 256 neurons
- **Activation function**: LeakyReLU (negative slope: 0.1)
- **Kernel initializer**: he_normal
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Dropout rate**: 0.3
- **Batch size**: 64
- **Max epochs**: 150

## 🚀 Getting Started

### Prerequisites
For this project, we use the Universal Virtual Environment (UV) to manage dependencies. Follow the installation guide here:
[uv install](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```

### Usage
1. Open the Jupyter notebook:
```bash
jupyter notebook fashion_mnist_analysis.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the Fashion MNIST dataset
   - Build and train the neural network
   - Analyze model performance
   - Generate visualizations

Or run the original Python script:
```bash
uv run main.py
```

## 📈 Training Features

### Advanced Callbacks
- **Early Stopping**: Stops training when validation accuracy plateaus (patience=30)
- **Learning Rate Scheduling**: Reduces learning rate by 50% when validation accuracy doesn't improve (patience=5)

### Data Preprocessing
- **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
- **No data augmentation**: Using original dataset as-is

## 🔍 Analysis Components

### 1. Activation Analysis
- Monitors neuron activation patterns
- Detects dying neurons (threshold: 1e-4)
- Analyzes LeakyReLU effectiveness

### 2. Training Visualization
- Training vs validation accuracy curves
- Training vs validation loss curves
- Epoch-by-epoch progress tracking

### 3. Per-Category Performance
- Individual accuracy for each fashion category
- Confidence intervals for accuracy estimates
- Best and worst performing categories

### 4. Model Health Metrics
- Dying neuron percentage
- Activation distribution analysis
- Statistical summaries of neuron behavior

## 📊 Generated Outputs

The notebook generates several visualization files:

| File | Description |
|------|-------------|
| `fashion_mnist_samples_raw.png` | Raw dataset samples (25 images) |
| `fashion_mnist_samples_normalized.png` | Normalized dataset samples |
| `training_history.png` | Training/validation accuracy and loss curves |
| `activation_distribution.png` | Distribution of neuron activations |
| `per_category_accuracy.png` | Per-category accuracy analysis |

## 🎯 Expected Results

- **Target accuracy**: ~90%
- **Dying neurons**: Zero or minimal (thanks to LeakyReLU)
- **Training stability**: Improved with learning rate scheduling
- **Clean output**: No matplotlib warnings

## 🧠 Key Optimizations Explained

### 1. Dying ReLU Problem Solution
**Problem**: Standard ReLU neurons can "die" when they always output zero.
**Solution**: LeakyReLU with negative slope of 0.1 allows small gradients for negative inputs.

### 2. Network Architecture Improvements
- **Larger hidden layer**: 256 neurons vs typical 128 for better capacity
- **Better initialization**: he_normal for improved gradient flow
- **Regularization**: 30% dropout to prevent overfitting

### 3. Training Optimizations
- **Learning rate scheduling**: Adaptive learning rate reduction
- **Early stopping**: Prevents overfitting with patience mechanism
- **Optimal batch size**: 64 for stable gradient updates

## 📁 Project Structure

```
ml/
├── fashion_mnist_analysis.ipynb    # Main Jupyter notebook
├── main.py                         # Original Python script
├── README.md                       # This file
├── pyproject.toml                  # Project configuration
├── uv.lock                         # Dependency lock file
└── generated_plots/                # Output visualizations
    ├── fashion_mnist_samples_raw.png
    ├── fashion_mnist_samples_normalized.png
    ├── training_history.png
    ├── activation_distribution.png
    └── per_category_accuracy.png
```

## 🔧 Hyperparameter Details

### LeakyReLU Configuration
- **negative_slope**: 0.1 (10% of input for negative values)
- **Purpose**: Prevents complete neuron death while maintaining non-linearity

### Training Parameters
- **patience (early stopping)**: 30 epochs
- **patience (learning rate)**: 5 epochs
- **learning rate reduction factor**: 0.5
- **minimum learning rate**: 1e-6

### Dying Neuron Detection
- **threshold**: 1e-4 (more lenient for LeakyReLU)
- **monitoring**: Maximum activation per neuron across all training data

## 📚 Learning Outcomes

This project demonstrates:
1. **Deep learning best practices** for image classification
2. **Activation function selection** and its impact on training
3. **Advanced training techniques** for better convergence
4. **Model analysis and visualization** for understanding performance
5. **Hyperparameter tuning** for optimal results

## 🤝 Contributing

Feel free to experiment with:
- Different activation functions
- Various network architectures
- Alternative optimization strategies
- Data augmentation techniques

## 📄 License

This project is for educational purposes. The Fashion-MNIST dataset is available under the MIT license.

---

**Note**: This implementation focuses on educational value and clear explanations rather than state-of-the-art performance. For production use, consider more advanced architectures like CNNs.
