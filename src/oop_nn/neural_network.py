import node, link
import math as m

class NeuralNetwork:

    """
    Initialize the neural network with input and output size

    Args:
        input_size (int): The number of input nodes
        output_size (int): The number of output nodes
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Create nodes and links
        self.nodes = []
        self.links = []
        self.create_nodes()
        self.create_links()

        # Store costs for each epoch
        self.costs = []

    
    """
    Create input and output nodes
    """
    def create_nodes(self):
        self.nodes.extend([node.Node() for _ in range(self.input_size + self.output_size)])

    """
    Create links between input and output nodes
    """
    def create_links(self):
        # Separate input and output nodes
        input_nodes = self.nodes[:self.input_size]
        output_nodes = self.nodes[-self.output_size:]

        # Clear existing links for each node
        for node in self.nodes:
            node.outgoing_links = []
            node.incoming_links = []

        # Create links and update incoming/outgoing links for nodes
        for input_node in input_nodes:
            for output_node in output_nodes:
                new_link = link.Link(input_node, output_node)
                self.links.append(new_link)
                input_node.outgoing_links.append(new_link)
                output_node.incoming_links.append(new_link)

    """
    Apply the sigmoid activation function to a value

    Args:
        x (float): A value

    Returns:
        float: The value after applying the sigmoid activation function
    """
    def sigmoid(self, x):
        return 1 / (1 + m.exp(-x))

    """
    Apply the softmax activation function to a list of values

    Args:
        x (list): A list of values

    Returns:
        list: A list of values after applying the softmax activation function
    """
    def softmax(self, x):
        exp_scores = [m.exp(i) for i in x]
        sum_exp_scores = sum(exp_scores)
        return [i / sum_exp_scores for i in exp_scores]
    
    """
    Perform a forward pass through the neural network

    Args:
        input_values (list): The input values for the input nodes
    
    Returns:
        list: The output values for the output nodes
    """
    def forward_pass(self, input_values):
        # Set input values
        for i, row in enumerate(input_values):
            for j, value in enumerate(row):
                # Calculate the index of the node in the input layer
                index = i * len(row) + j  
                self.nodes[index].value = value

        # Calculate the sum for all nodes except input nodes
        for node in self.nodes[-self.output_size:]:
            node.value = node.calculate_sum()

        # Apply softmax activation function for output nodes
        softmax_values = self.softmax([node.value for node in self.nodes[-self.output_size:]])
        # sigmoid_values = [self.sigmoid(node.value) for node in self.nodes[-self.output_size:]]

        # Update values for output nodes with softmax values
        for i, node in enumerate(self.nodes[-self.output_size:]):
            node.value = softmax_values[i]
            # node.value = sigmoid_values[i]

        # Return output values and print them
        output_values = [node.value for node in self.nodes[-self.output_size:]]
        return output_values

    """
    Calculate the mean squared error cost for a given input and target values

    Args:
        input_values (list): The input values for the input nodes
        target_values (list): The target values for the output nodes

    Returns:
        float: The mean squared error cost
    """
    def mse_cost(self, input_values, target_values):
        # Calculate output values
        output_values = self.forward_pass(input_values)

        if target_values == 'O':
            target_values = [1, 0]
        else:
            target_values = [0, 1]
        
        # Calculate mean squared error
        squared_error = 0
        for i in range(len(target_values)):
            squared_error += (target_values[i] - output_values[i]) ** 2
        
        mse = squared_error / len(target_values)

        return mse
    
    """
    Perform backpropagation to calculate gradients for weights and biases

    Args:
        target_values (list): The target values for the output nodes

    Returns:
        dict: A dictionary containing the gradients for each link and node
    """
    def back_propagation(self, target_values):
        if target_values == 'O':
            target_values = [1, 0]
        else:
            target_values = [0, 1]

        # Calculate gradients
        gradients = {}
        for link in self.links:
            gradients[link] = {'weight': 0}
        for node in self.nodes[-self.output_size:]:
            gradients[node] = {'bias': 0}

        # Calculate gradients for output nodes
        for i, node in enumerate(self.nodes[-self.output_size:]):
            # Calculate bias gradient
            # The derivative of an activation function is node.value * (1 - node.value)
            # The derivative of the mean squared error cost is 2 * (node.value - target_values[i])
            derivative_mse = 2 * (node.value - target_values[i])
            derivative_activation = node.value * (1 - node.value)
            gradients[node]['bias'] = derivative_mse * derivative_activation
            for link in node.incoming_links:
                # Calculate weight gradient
                # The gradient for the weight of a link is the gradient of the bias multiplied by the value of the node that the link is coming from
                derivative_output = link.from_node.value
                gradients[link]['weight'] = derivative_mse * derivative_output * derivative_activation
    
        return gradients

    """
    Update the weights and biases of the neural network using the calculated gradients

    Args:
        learning_rate (float): The learning rate for updating weights and biases
        gradient (dict): A dictionary containing the gradients for each link and node
    """
    def update_wb(self, learning_rate, gradient):
        for link in self.links:
            link.weight -= learning_rate * gradient[link]['weight']
        for node in self.nodes[-self.output_size:]:
            node.bias -= learning_rate * gradient[node]['bias']

    """
    Train the neural network using backpropagation
    
    Args:
        training_set (list): A list of tuples containing input and target values
        max_cost (float): The maximum mean squared error cost
        learning_rate (float): The learning rate for updating weights and biases
        
    """
    def train(self, training_set, max_cost, learning_rate):
        cost = max_cost + 1
        epoch = 0   
        while cost > max_cost:
            cost = 0
            for input_values, target_values in training_set:
                cost += self.mse_cost(input_values, target_values)
                gradient = self.back_propagation(target_values)
                self.update_wb(learning_rate, gradient)
            cost /= len(training_set)
            self.costs.append(cost)
            epoch += 1

        print(f'Epoch: {epoch} Cost: {cost}')

    """
    Predict the output values for a given test set

    Args:
        test_set (list): A list of tuples containing input and target values 
    """
    def predict(self, test_set):
        predictions = []
        for input_values, target_values in test_set:
            output_values = self.forward_pass(input_values)
            # Choose the output node with the highest value
            if output_values[0] > output_values[1]:
                # 'O' represents the first output node
                predictions.append('O')
            else:
                # 'X' represents the second output node
                predictions.append('X')
        return predictions
    
    """
    Calculate the accuracy of the neural network on a given test set

    Args:
        test_set (list): A list of tuples containing input and target values
    """
    def accuracy(self, test_set):
        predictions = self.predict(test_set)
        correct = 0
        for i in range(len(predictions)):
            print(f'Prediction: {predictions[i]} Label: {test_set[i][1]}')
            if predictions[i] == test_set[i][1]:
                correct += 1
        accuracy = correct / len(test_set) * 100
        return f'Accuracy: {accuracy}%'