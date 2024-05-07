import node, link, data as dt
import math as m

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        """
        Initialize the neural network with input and output size

        Args:
            input_size (int): The number of input nodes
            output_size (int): The number of output nodes
        """
        
        self.input_size = input_size
        self.output_size = output_size

        # Create nodes and links
        self.nodes = []
        self.links = []
        self.create_nodes()
        self.create_links()

        # Store costs for each epoch
        self.costs = []
    
    
    def create_nodes(self):
        """
        Create input and output nodes
        """
        self.nodes.extend([node.Node() for _ in range(self.input_size + self.output_size)])

    
    def create_links(self):
        """
        Create links between input and output nodes
        """
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


    def sigmoid(self, x):
        """
        Apply the sigmoid activation function to a value

        Args:
            x (float): A value

        Returns:
            float: The value after applying the sigmoid activation function
        """
        return 1 / (1 + m.exp(-x))


    def softmax(self, x):
        """
        Apply the softmax activation function to a list of values

        Args:
            x (list): A list of values

        Returns:
            list: A list of values after applying the softmax activation function
        """
        exp_scores = [m.exp(i) for i in x]
        sum_exp_scores = sum(exp_scores)
        return [i / sum_exp_scores for i in exp_scores]
    
    
    def forward_pass(self, input_values):
        """
        Perform a forward pass through the neural network

        Args:
            input_values (list): The input values for the input nodes
        
        Returns:
            list: The output values for the output nodes
        """
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

   
    def mse_cost(self, input_values, target_values):
        """
        Calculate the mean squared error cost for a given input and target values

        Args:
            input_values (list): The input values for the input nodes
            target_values (list): The target values for the output nodes

        Returns:
            float: The mean squared error cost
        """
        # Calculate output values
        output_values = self.forward_pass(input_values)

        target_values = dt.outputDict[target_values]
        
        # Calculate mean squared error
        squared_error = 0
        for i in range(len(target_values)):
            squared_error += (target_values[i] - output_values[i]) ** 2
        
        mse = squared_error / len(target_values)

        return mse
    

    def back_propagation(self, target_values, learning_rate):
        """
        Perform backpropagation to update weights and biases

        Args:
            target_values (list): The target values for the output nodes
            learning_rate (float): The learning rate for updating weights and biases

        Returns:
            None
        """
        target_values = dt.outputDict[target_values]

        # Calculate gradients and update weights and biases
        for i, node in enumerate(self.nodes[-self.output_size:]):
            # Calculate bias gradient for output nodes
            derivative_mse = 2 * (node.value - target_values[i])
            derivative_activation = node.value * (1 - node.value)
            node.bias -= learning_rate * derivative_mse * derivative_activation

            # Update weights for incoming links
            for link in node.incoming_links:
                derivative_output = link.from_node.value
                link.weight -= learning_rate * derivative_mse * derivative_output * derivative_activation



    def train(self, training_set, max_cost, learning_rate):
        """
        Train the neural network using backpropagation
        
        Args:
            training_set (list): A list of tuples containing input and target values
            max_cost (float): The maximum mean squared error cost
            learning_rate (float): The learning rate for updating weights and biases
            
        """
        cost = max_cost + 1
        epoch = 0   
        while cost > max_cost:
            cost = 0
            for input_values, target_values in training_set:
                cost += self.mse_cost(input_values, target_values)
                self.back_propagation(target_values, learning_rate)
            cost /= len(training_set)
            self.costs.append(cost)
            epoch += 1

        print(f'Epoch: {epoch} Cost: {cost}')


    def predict(self, test_set):
        """
        Predict the output values for a given test set

        Args:
            test_set (list): A list of tuples containing input and target values 
        """
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
    
    def accuracy(self, test_set):
        """
        Calculate the accuracy of the neural network on a given test set

        Args:
            test_set (list): A list of tuples containing input and target values
        """
        predictions = self.predict(test_set)
        correct = 0
        for i in range(len(predictions)):
            print(f'Prediction: {predictions[i]} Label: {test_set[i][1]}')
            if predictions[i] == test_set[i][1]:
                correct += 1
        accuracy = correct / len(test_set) * 100
        return f'Accuracy: {accuracy}%'