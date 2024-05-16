import numpy as np
import data as dt

class NeuralNetwork:
    """
    A simple feedforward neural network implemented with numpy.

    Attributes:
        layer_sizes (list): The number of neurons in each layer of the network.
        num_layers (int): The number of layers in the network.
        weights (list): List of weight matrices for each layer.
        biases (list): List of bias vectors for each layer.
        costs (list): List to store the cost of the network during training.
        accuracies (list): List to store the accuracy of the network during training.
    """

    def __init__(self, layer_sizes: list):
        """
        Initializes the neural network with random weights and biases.

        Args:
            layer_sizes (list): The number of neurons in each layer of the network.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases between -1 and 1
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i + 1])) for i in range(self.num_layers - 1)]
        self.biases = [np.random.uniform(-1, 1, (1, layer_sizes[i + 1])) for i in range(self.num_layers - 1)]
        self.costs = []
        self.accuracies = []

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid activation function element-wise.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the sigmoid activation function.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Derivative of the sigmoid function.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the network.

        Args:
            inputs (ndarray): Input data.

        Returns:
            ndarray: Output of the network.
        """
        self.layer_outputs = [inputs]
        for i in range(self.num_layers - 1):
            layer_input = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)
        return self.layer_outputs[-1]

    def mean_squared_error(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Computes the mean squared error between predictions and labels.

        Args:
            predictions (ndarray): Predicted values.
            labels (ndarray): True labels.

        Returns:
            float: Mean squared error.
        """
        return np.mean((predictions - labels) ** 2)

    def backpropagation(self, predictions: np.ndarray, labels: np.ndarray, learning_rate: float = 0.1):
        """
        Performs backpropagation to update weights and biases.

        Args:
            predictions (ndarray): Predicted values.
            labels (ndarray): True labels.
            learning_rate (float): Learning rate for updating weights and biases.
        """
        # Calculate the error for the output layer with chain rule applied to the derivative of the sigmoid function and the mean squared error.
        output_error = 2 * (predictions - labels) * self.sigmoid_derivative(self.layer_outputs[-1])
        
        # Iterate over the layers in reverse order
        for i in range(self.num_layers - 2, -1, -1):
            # Calculate the error for the current layer
            layer_error = np.dot(output_error, self.weights[i].T)

            # Calculate the gradient of the loss with respect to the weights
            weight_gradient = np.dot(self.layer_outputs[i].T, output_error)

            # Calculate the gradient of the loss with respect to the biases
            bias_gradient = np.sum(output_error, axis=0)

            # Update the weights and biases
            self.weights[i] -= learning_rate * weight_gradient
            self.biases[i] -= learning_rate * bias_gradient
            
            # Update the error for the next iteration
            output_error = layer_error

    def train(self, inputs: np.ndarray, labels: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1):
        """
        Trains the neural network using backpropagation.

        Args:
            inputs (ndarray): Input data.
            labels (ndarray): True labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for updating weights and biases.
        """
        for epoch in range(epochs):
            predictions = self.forward_propagation(inputs)
            cost = self.mean_squared_error(predictions, labels)
            self.costs.append(cost)
            accuracy = self.accuracy(inputs, labels)
            self.accuracies.append(accuracy)
            if cost < dt.maxCost:
                break
            self.backpropagation(predictions, labels, learning_rate)
        self.print_predictions(inputs, labels)
        print(f"Cost: {cost} epochs: {epoch + 1} Accuracy: {accuracy}%")


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained network.

        Args:
            inputs (ndarray): Input data.

        Returns:
            ndarray: Predicted labels.
        """
        predictions = self.forward_propagation(inputs)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels

    def accuracy(self, inputs: np.ndarray, labels: np.ndarray) -> str:
        """
        Computes the accuracy of the network on a given dataset.

        Args:
            inputs (ndarray): Input data.
            labels (ndarray): True labels.

        Returns:
            str: Accuracy percentage.
        """
        predictions = self.predict(inputs)
        correct = np.sum(predictions == np.argmax(labels, axis=1))
        accuracy = (correct / len(labels)) * 100
        return accuracy
    
    def print_predictions(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Prints predictions after training.

        Args:
            inputs (ndarray): Input data.
            labels (ndarray): True labels.
        """
        predictions = self.predict(inputs)
        for i, predicted_label_index in enumerate(predictions):
            predicted_label = list(dt.outputDict.keys())[predicted_label_index]
            true_label = list(dt.outputDict.keys())[np.argmax(labels[i])]
            print(f"Prediction: {predicted_label} Label: {true_label}")