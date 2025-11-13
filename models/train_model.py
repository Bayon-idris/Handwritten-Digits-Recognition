import numpy as np

class BPNetwork:
    """
    Simple 3-layer BP Neural Network
    Input layer: 256 nodes
    Hidden layer: q hidden nodes
    Output layer: 10 nodes
    """

    def __init__(self, input_size=256, hidden_size=25, output_size=10, lr=0.1, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs

        # Randomly initialize weights
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, a):
        return a * (1 - a)

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = Z2  # linear output for regression, but for classification can also use sigmoid/softmax
        return Z1, A1, Z2, A2

    def compute_loss(self, Y_pred, Y_true):
        return np.mean((Y_pred - Y_true)**2)

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_deriv(A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y):
        for epoch in range(self.epochs):
            Z1, A1, Z2, A2 = self.forward(X)
            loss = self.compute_loss(A2, Y)
            self.backward(X, Y, Z1, A1, Z2, A2)
            if (epoch+1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.6f}")

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0)
