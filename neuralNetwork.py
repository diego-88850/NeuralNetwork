import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import double
from tensorflow.keras.datasets import mnist
import pickle


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss function"""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def deriv_relu(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


class OurNeuralNetwork:
    def __init__(self, layers_sizes, learning_rate, l2_lambda):
        self.lr = learning_rate
        self.num_layers = len(layers_sizes) - 1
        self.weights = []
        self.biases = []
        self.l2_lambda = l2_lambda

        for i in range(self.num_layers):
            in_dim = layers_sizes[i]
            out_dim = layers_sizes[i + 1]

            # He initialization for ReLU layers
            if i < self.num_layers - 1:  # Hidden layers
                W = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
            else:  # Output layer
                W = np.random.randn(out_dim, in_dim) * np.sqrt(1 / in_dim)

            b = np.zeros(out_dim)  # Initialize biases to zero
            self.weights.append(W)
            self.biases.append(b)

    def feedforward(self, X):
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
        m = y_true.shape[0]

        # For softmax + cross-entropy, the gradient is simply (y_pred - y_true)
        delta = (activ[-1] - y_true) / m

        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

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
        activ, pre_activ = self.feedforward(x_batch)
        y_pred = activ[-1]
        loss = cross_entropy_loss(y_batch, y_pred)

        self.backprop(activ, pre_activ, y_batch)
        return loss

    def fit(self, X, Y, epochs, batch_size, verbose):
        n = len(X)
        losses = []

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n)
            X, Y = X[perm], Y[perm]

            epoch_loss = 0.0

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


def one_hot(y, num_classes):
    m = y.shape[0]
    ret = np.zeros((m, num_classes))
    ret[np.arange(m), y] = 1
    return ret


class Settings:
    def __init__(self, learning_rate, epochs, batch_size, verbose, l2_lambda):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.l2_lambda = l2_lambda


def get_user_hyperparameters():
    try:
        lr = float(input("Enter learning rate (should be pretty small): "))
    except ValueError:
        print("⚠️ Invalid input. Using default learning rate of 0.1.")
        lr = 0.1
    try:
        eps = int(input("Enter number of epochs: "))
    except ValueError:
        print("<UNK> Invalid input. Using default epochs = 20.")
        eps = 20
    try:
        bs = int(input("Enter batch size: "))
    except ValueError:
        print("<UNK> Invalid input. Using default batch size = 64.")
        bs = 64

    l2 = (input("Do you want to regularize the weights? (y/n): "))
    if l2 == "y":
        try:
            l2 = float(input("Enter L2 regularization coefficient: "))
        except ValueError:
            print("<UNK> Invalid input. Using default L2 regularization coefficient of 0.001.")
            l2 = 0.001
    else:
        l2 = 0

    temp = input("Verbose mode y/n: ")
    verb = True if temp == "y" else False

    settings = Settings(lr, eps, bs, verb, l2)

    num_layers = int(
        input("Enter the number of hidden layers between 1 and 7 OR type 42 for default ([784, 128, 64, 10]): "))
    if num_layers == 42:
        return [784, 128, 64, 10], settings
    if num_layers < 1 or num_layers > 7:
        print("Invalid number of layers, submitting default")
        return [784, 128, 64, 10], settings

    res = [784]
    for i in range(num_layers):
        layer_size = int(input("choose layer size: "))
        res.append(layer_size)
    res.append(10)
    return res, settings


def predict(xtr, ytr, xts, yts):
    xtr = xtr.reshape(-1, 28 * 28) / 255.0
    xts = xts.reshape(-1, 28 * 28) / 255.0
    ytr = one_hot(ytr, 10)
    yts = one_hot(yts, 10)

    res, settings = get_user_hyperparameters()
    net = OurNeuralNetwork(res, learning_rate=settings.learning_rate, l2_lambda=settings.l2_lambda)
    losses = net.fit(xtr, ytr, epochs=settings.epochs, batch_size=settings.batch_size, verbose=settings.verbose)

    # matplotlib graphs for each of the data
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (Cross-Entropy)")
    plt.title("Training Loss vs. Epoch")
    plt.show()

    activates, _ = net.feedforward(xts)
    y_pred_probs = activates[-1]
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(yts, axis=1)

    accuracy = np.sum(y_pred_labels == y_true_labels) / len(y_true_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    size = int(input("How many random tests do you want?: "))
    test_indices = np.random.randint(0, 10000, size=size).tolist()

    for idx in test_indices:
        x = xts[idx]

        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Index {idx}  —  True: {y_test[idx]}")
        plt.axis("off")
        plt.show()

        activs, _ = net.feedforward(x[np.newaxis, :])
        probs = activs[-1]
        pred_label = np.argmax(probs, axis=1)[0]
        confidence = probs[0, pred_label]

        print(f"→ Predicted {pred_label}, True {y_test[idx]} (confidence {confidence:.2f})\n")


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    predict(X_train, y_train, X_test, y_test)