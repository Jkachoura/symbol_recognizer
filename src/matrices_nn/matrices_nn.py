import numpy as np
import data as dt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights number between -1 and 1
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        
        # Initialize biases number between -1 and 1
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        
        self.costs = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activation_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward_propagation(self, inputs):
        self.layer_outputs = [inputs]
        for i in range(self.num_layers - 1):
            layer_input = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)
        
        return self.layer_outputs[-1]

    def mean_squared_error(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)

    def backpropagation(self, inputs, predictions, labels, learning_rate=0.1):
        output_error = 2* (predictions - labels)
        deltas = [output_error * self.activation_derivative(predictions)]
        
        for i in range(self.num_layers - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.activation_derivative(self.layer_outputs[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * self.layer_outputs[i].T.dot(deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, inputs, labels, max_cost=0.01,  learning_rate=0.01):
        loss = 1
        epoch = 0
        while loss > max_cost:
            predictions = self.forward_propagation(inputs)
            loss = self.mean_squared_error(predictions, labels)
            self.costs.append(loss)
            self.backpropagation(inputs, predictions, labels, learning_rate)
            epoch += 1
        
        # cost is  and epoch is print
        print(f"Epoch: {epoch} Cost: {loss}")
    
    def predict(self, inputs):
        predictions = self.forward_propagation(inputs)
        predicted_labels = np.argmax(predictions, axis=1)
        for i, predicted_label_index in enumerate(predicted_labels):
            predicted_label = list(dt.outputDict.keys())[predicted_label_index]
            true_label = list(dt.outputDict.keys())[np.argmax(inputs[i][-1])]  # Accessing the last row for true label
            print(f"Prediction: {predicted_label} Label: {true_label}")
        return predicted_labels
    
    def accuracy(self, inputs, labels):
        predictions = self.predict(inputs)
        correct = np.sum(predictions == np.argmax(labels, axis=1))
        accuracy = (correct / len(labels)) * 100
        return f"Accuracy: {accuracy}%"

