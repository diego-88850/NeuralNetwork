import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import pickle


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    """ReLU activation function: f(x) = max(0, x)"""
    return np.maximum(0, x)


def softmax(x):
    """Softmax activation function with numerical stability"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def deriv_relu(x):
    """Derivative of ReLU function"""
    return (x > 0).astype(float)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss function"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def one_hot(y, num_classes):
    """Convert labels to one-hot encoding"""
    m = y.shape[0]
    ret = np.zeros((m, num_classes))
    ret[np.arange(m), y] = 1
    return ret


def save_model(model, filename):
    """Save a trained model to disk"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved successfully as '{filename}'")
    except Exception as e:
        print(f"❌ Error saving model: {e}")


def load_model(filename):
    """Load a trained model from disk"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from '{filename}'")
        return model
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def list_saved_models():
    """List all saved model files in current directory"""
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if model_files:
        print("📁 Saved models:")
        for i, filename in enumerate(model_files, 1):
            print(f"  {i}. {filename}")
    else:
        print("📁 No saved models found.")
    return model_files


# =============================================================================
# SETTINGS CLASS
# =============================================================================

class Settings:
    """Class to store training hyperparameters"""

    def __init__(self, learning_rate, epochs, batch_size, verbose, l2_lambda):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.l2_lambda = l2_lambda


# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================

class OurNeuralNetwork:
    """Multi-layer neural network with ReLU hidden layers and Softmax output"""

    def __init__(self, layers_sizes, learning_rate, l2_lambda):
        """
        Initialize neural network

        Args:
            layers_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            learning_rate: Learning rate for gradient descent
            l2_lambda: L2 regularization coefficient
        """
        self.lr = learning_rate
        self.num_layers = len(layers_sizes) - 1
        self.weights = []
        self.biases = []
        self.l2_lambda = l2_lambda

        # Initialize weights and biases
        for i in range(self.num_layers):
            in_dim = layers_sizes[i]
            out_dim = layers_sizes[i + 1]

            # He initialization for ReLU layers, Xavier for output
            if i < self.num_layers - 1:  # Hidden layers
                W = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
            else:  # Output layer
                W = np.random.randn(out_dim, in_dim) * np.sqrt(1 / in_dim)

            b = np.zeros(out_dim)  # Initialize biases to zero
            self.weights.append(W)
            self.biases.append(b)

    def feedforward(self, X):
        """
        Forward pass through the network

        Args:
            X: Input data

        Returns:
            activations: List of activations for each layer
            pre_activations: List of pre-activations for each layer
        """
        A = X
        activ = [X]
        pre_activ = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A.dot(W.T) + b
            pre_activ.append(Z)

            # Apply activation function
            if i < self.num_layers - 1:  # Hidden layers use ReLU
                A = relu(Z)
            else:  # Output layer uses softmax
                A = softmax(Z)

            activ.append(A)

        return activ, pre_activ

    def backprop(self, activ, pre_activ, y_true):
        """
        Backward pass (backpropagation)

        Args:
            activ: Activations from forward pass
            pre_activ: Pre-activations from forward pass
            y_true: True labels (one-hot encoded)
        """
        m = y_true.shape[0]

        # For softmax + cross-entropy, gradient is (y_pred - y_true) / m
        delta = (activ[-1] - y_true) / m

        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Compute gradients layer by layer (backwards)
        for l in reversed(range(self.num_layers)):
            # Compute gradients
            grads_W[l] = delta.T @ activ[l] + self.l2_lambda * self.weights[l]
            grads_b[l] = delta.sum(axis=0)

            # Compute delta for next layer (if not the first layer)
            if l > 0:
                # Backpropagate through ReLU
                delta = delta.dot(self.weights[l]) * deriv_relu(pre_activ[l - 1])

        # Update weights and biases
        for l in range(self.num_layers):
            self.weights[l] -= self.lr * grads_W[l]
            self.biases[l] -= self.lr * grads_b[l]

    def train(self, x_batch, y_batch):
        """
        Train on a single batch

        Args:
            x_batch: Input batch
            y_batch: Output batch (one-hot encoded)

        Returns:
            loss: Cross-entropy loss for this batch
        """
        activ, pre_activ = self.feedforward(x_batch)
        y_pred = activ[-1]
        loss = cross_entropy_loss(y_batch, y_pred)

        self.backprop(activ, pre_activ, y_batch)
        return loss

    def fit(self, X, Y, epochs, batch_size, verbose):
        """
        Train the neural network

        Args:
            X: Training data
            Y: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress

        Returns:
            losses: List of losses for each epoch
        """
        n = len(X)
        losses = []

        for epoch in range(1, epochs + 1):
            # Shuffle data
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, n, batch_size):
                x_batch = X[i: i + batch_size]
                y_batch = Y[i: i + batch_size]
                batch_loss = self.train(x_batch, y_batch)
                epoch_loss += batch_loss * len(x_batch)

            epoch_loss /= n

            if verbose:
                print(f"Epoch {epoch:3d} loss = {epoch_loss:.4f}")

            losses.append(epoch_loss)

        return losses


# =============================================================================
# MODEL MANAGER CLASS
# =============================================================================

class ModelManager:
    """Manages multiple trained models for comparison"""

    def __init__(self):
        self.models = {}  # Dictionary to store loaded models
        self.model_info = {}  # Store metadata about each model

    def add_model(self, name, model, settings, test_accuracy, losses):
        """Add a trained model to the manager"""
        self.models[name] = model
        self.model_info[name] = {
            'settings': settings,
            'test_accuracy': test_accuracy,
            'final_loss': losses[-1] if losses else None,
            'total_epochs': len(losses)
        }
        print(f"✅ Model '{name}' added to manager")

    def save_model(self, name, filename=None):
        """Save a specific model"""
        if name not in self.models:
            print(f"❌ Model '{name}' not found in manager")
            return False

        if filename is None:
            filename = f"{name}.pkl"

        # Save both model and its metadata
        model_data = {
            'model': self.models[name],
            'info': self.model_info[name]
        }

        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model '{name}' saved as '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, filename, name=None):
        """Load a model from file"""
        if name is None:
            name = filename.replace('.pkl', '')

        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            # Handle both old format (just model) and new format (model + info)
            if isinstance(model_data, dict) and 'model' in model_data:
                self.models[name] = model_data['model']
                self.model_info[name] = model_data.get('info', {})
            else:
                # Old format - just the model
                self.models[name] = model_data
                self.model_info[name] = {'legacy_model': True}

            print(f"✅ Model loaded as '{name}'")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def list_models(self):
        """List all models in manager"""
        if not self.models:
            print("📂 No models in manager")
            return

        print("📂 Models in manager:")
        print("-" * 60)
        for name, info in self.model_info.items():
            accuracy = info.get('test_accuracy', 'Unknown')
            final_loss = info.get('final_loss', 'Unknown')
            epochs = info.get('total_epochs', 'Unknown')

            if isinstance(accuracy, float):
                accuracy = f"{accuracy * 100:.2f}%"
            if isinstance(final_loss, float):
                final_loss = f"{final_loss:.4f}"

            print(f"  🤖 {name}")
            print(f"     Accuracy: {accuracy} | Final Loss: {final_loss} | Epochs: {epochs}")

    def compare_models(self, model_names=None):
        """Compare multiple models"""
        if model_names is None:
            model_names = list(self.models.keys())

        if len(model_names) < 2:
            print("❌ Need at least 2 models to compare")
            return

        print("📊 Model Comparison:")
        print("=" * 80)

        # Sort by accuracy for better comparison
        valid_models = []
        for name in model_names:
            if name in self.models:
                info = self.model_info[name]
                accuracy = info.get('test_accuracy', 0)
                valid_models.append((name, accuracy, info))

        valid_models.sort(key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)

        for i, (name, accuracy, info) in enumerate(valid_models, 1):
            final_loss = info.get('final_loss', 'Unknown')
            epochs = info.get('total_epochs', 'Unknown')
            settings = info.get('settings')

            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

            print(f"{rank_emoji} {name}")
            if isinstance(accuracy, float):
                print(f"   📈 Accuracy: {accuracy * 100:.2f}%")
            if isinstance(final_loss, float):
                print(f"   📉 Final Loss: {final_loss:.4f}")
            print(f"   🔄 Epochs: {epochs}")

            if settings:
                print(
                    f"   ⚙️  Settings: LR={settings.learning_rate}, BS={settings.batch_size}, L2={settings.l2_lambda}")
            print()

    def get_model(self, name):
        """Get a specific model"""
        return self.models.get(name)

    def remove_model(self, name):
        """Remove a model from manager"""
        if name in self.models:
            del self.models[name]
            del self.model_info[name]
            print(f"✅ Model '{name}' removed from manager")
        else:
            print(f"❌ Model '{name}' not found")


# =============================================================================
# USER INPUT FUNCTIONS
# =============================================================================

def get_user_hyperparameters():
    """Get hyperparameters from user input"""
    print("\n⚙️ Configure Training Parameters")
    print("-" * 35)

    # Learning rate
    try:
        lr = float(input("Enter learning rate (should be pretty small): "))
    except ValueError:
        print("⚠️ Invalid input. Using default learning rate of 0.1.")
        lr = 0.1

    # Epochs
    try:
        eps = int(input("Enter number of epochs: "))
    except ValueError:
        print("⚠️ Invalid input. Using default epochs = 20.")
        eps = 20

    # Batch size
    try:
        bs = int(input("Enter batch size: "))
    except ValueError:
        print("⚠️ Invalid input. Using default batch size = 64.")
        bs = 64

    # L2 regularization
    l2 = input("Do you want to regularize the weights? (y/n): ")
    if l2 == "y":
        try:
            l2 = float(input("Enter L2 regularization coefficient: "))
        except ValueError:
            print("⚠️ Invalid input. Using default L2 regularization coefficient of 0.001.")
            l2 = 0.001
    else:
        l2 = 0

    # Verbose mode
    temp = input("Verbose mode y/n: ")
    verb = True if temp == "y" else False

    settings = Settings(lr, eps, bs, verb, l2)

    # Network architecture
    print("\n🏗️ Configure Network Architecture")
    print("-" * 35)
    num_layers = int(
        input("Enter the number of hidden layers between 1 and 7 OR type 42 for default ([784, 128, 64, 10]): "))

    if num_layers == 42:
        return [784, 128, 64, 10], settings
    if num_layers < 1 or num_layers > 7:
        print("Invalid number of layers, using default")
        return [784, 128, 64, 10], settings

    res = [784]  # Input layer (28x28 = 784 pixels)
    for i in range(num_layers):
        layer_size = int(input(f"Enter size for hidden layer {i + 1}: "))
        res.append(layer_size)
    res.append(10)  # Output layer (10 classes for digits)

    return res, settings


# =============================================================================
# MENU FUNCTIONS
# =============================================================================

def train_new_model(manager):
    """Train a new model and add it to manager"""
    print("\n🎯 Training New Model")
    print("-" * 30)

    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    y_train_onehot = one_hot(y_train, 10)

    # Get model configuration
    layers, settings = get_user_hyperparameters()

    # Get model name
    model_name = input("\nEnter a name for this model: ").strip()
    if not model_name:
        model_name = f"model_{len(manager.models) + 1}"

    # Train model
    print(f"\n🏋️ Training model '{model_name}'...")
    net = OurNeuralNetwork(layers, learning_rate=settings.learning_rate, l2_lambda=settings.l2_lambda)
    losses = net.fit(X_train, y_train_onehot, epochs=settings.epochs, batch_size=settings.batch_size,
                     verbose=settings.verbose)

    # Evaluate model
    activates, _ = net.feedforward(X_test)
    y_pred_probs = activates[-1]
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    accuracy = np.sum(y_pred_labels == y_test) / len(y_test)

    print(f"✅ Training complete! Test accuracy: {accuracy * 100:.2f}%")

    # Add to manager
    manager.add_model(model_name, net, settings, accuracy, losses)

    # Show training plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (Cross-Entropy)")
    plt.title(f"Training Loss vs. Epoch - {model_name}")
    plt.grid(True)
    plt.show()


def load_model_menu(manager):
    """Load a model from disk"""
    print("\n💾 Load Saved Model")
    print("-" * 20)

    # Show available models
    model_files = list_saved_models()
    if not model_files:
        return

    filename = input("\nEnter filename to load: ").strip()
    if filename and not filename.endswith('.pkl'):
        filename += '.pkl'

    model_name = input("Enter name for loaded model (press Enter for default): ").strip()

    if manager.load_model(filename, model_name if model_name else None):
        print("Model loaded successfully!")


def compare_models_menu(manager):
    """Compare models in manager"""
    print("\n📊 Compare Models")
    print("-" * 17)

    if len(manager.models) < 2:
        print("❌ Need at least 2 models in manager to compare")
        return

    manager.list_models()
    print("\nLeave empty to compare all models, or enter model names separated by commas:")
    model_input = input("Models to compare: ").strip()

    if model_input:
        model_names = [name.strip() for name in model_input.split(",")]
        manager.compare_models(model_names)
    else:
        manager.compare_models()


def test_model_menu(manager):
    """Test a specific model with random samples"""
    print("\n🔍 Test Individual Model")
    print("-" * 25)

    if not manager.models:
        print("❌ No models in manager")
        return

    manager.list_models()
    model_name = input("\nEnter model name to test: ").strip()

    model = manager.get_model(model_name)
    if model is None:
        print(f"❌ Model '{model_name}' not found")
        return

    # Load test data
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    size = int(input("How many random tests do you want?: "))
    test_indices = np.random.randint(0, 10000, size=size).tolist()

    for idx in test_indices:
        x = X_test[idx]

        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Model: {model_name} | Index {idx} | True: {y_test[idx]}")
        plt.axis("off")
        plt.show()

        activs, _ = model.feedforward(x[np.newaxis, :])
        probs = activs[-1]
        pred_label = np.argmax(probs, axis=1)[0]
        confidence = probs[0, pred_label]

        print(f"→ Predicted {pred_label}, True {y_test[idx]} (confidence {confidence:.2f})\n")


def save_model_menu(manager):
    """Save a model from manager to disk"""
    print("\n💾 Save Model")
    print("-" * 12)

    if not manager.models:
        print("❌ No models in manager")
        return

    manager.list_models()
    model_name = input("\nEnter model name to save: ").strip()

    if model_name not in manager.models:
        print(f"❌ Model '{model_name}' not found")
        return

    filename = input("Enter filename (press Enter for default): ").strip()
    if filename and not filename.endswith('.pkl'):
        filename += '.pkl'

    manager.save_model(model_name, filename if filename else None)


def remove_model_menu(manager):
    """Remove a model from manager"""
    print("\n🗑️ Remove Model")
    print("-" * 14)

    if not manager.models:
        print("❌ No models in manager")
        return

    manager.list_models()
    model_name = input("\nEnter model name to remove: ").strip()
    manager.remove_model(model_name)


def main_menu():
    """Interactive menu system"""
    manager = ModelManager()

    while True:
        print("\n" + "=" * 50)
        print("🧠 Neural Network Training & Comparison Tool")
        print("=" * 50)
        print("1. 🎯 Train new model")
        print("2. 💾 Load saved model")
        print("3. 📂 List saved models on disk")
        print("4. 📋 List models in manager")
        print("5. 📊 Compare models")
        print("6. 🔍 Test individual model")
        print("7. 💾 Save model from manager")
        print("8. 🗑️  Remove model from manager")
        print("9. 🚪 Exit")

        choice = input("\nEnter your choice (1-9): ").strip()

        if choice == "1":
            train_new_model(manager)
        elif choice == "2":
            load_model_menu(manager)
        elif choice == "3":
            list_saved_models()
        elif choice == "4":
            manager.list_models()
        elif choice == "5":
            compare_models_menu(manager)
        elif choice == "6":
            test_model_menu(manager)
        elif choice == "7":
            save_model_menu(manager)
        elif choice == "8":
            remove_model_menu(manager)
        elif choice == "9":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    main_menu()