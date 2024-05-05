import node, link

class NeuralNetwork:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = []
        self.links = []
        self.create_nodes()
        self.create_links()
        self.costs = []

    def create_nodes(self):
        for i in range(self.input_size):
            self.nodes.append(node.Node())
        for i in range(self.output_size):
            self.nodes.append(node.Node())

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

    def forward_pass(self, input_values):
        # Set input values
        for i, row in enumerate(input_values):
            for j, value in enumerate(row):
                index = i * len(row) + j  # Calculate the index in a flattened manner
                self.nodes[index].value = value

        # Calculate values for all nodes except input nodes
        for node in self.nodes[-self.output_size:]:
            node.calculate_value()

        # Return output values
        return [node.value for node in self.nodes[-self.output_size:]]
    
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
    
    
    def back_propagation(self, input_values, target_values):
        # Calculate output values
        output_values = self.forward_pass(input_values)

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
            gradients[node]['bias'] = (node.value - target_values[i]) * node.value * (1 - node.value)
            for link in node.incoming_links:
                # Calculate weight gradient
                gradients[link]['weight'] = gradients[node]['bias'] * link.from_node.value
    
        return gradients

    def update_wb(self, learning_rate, gradient):
        for link in self.links:
            link.weight -= learning_rate * gradient[link]['weight']
        for node in self.nodes[-self.output_size:]:
            node.bias -= learning_rate * gradient[node]['bias']

    def train(self, training_set, max_cost, learning_rate):
        cost = max_cost + 1
        epoch = 0   
        while cost > max_cost:
            cost = 0
            for input_values, target_values in training_set:
                cost += self.mse_cost(input_values, target_values)
                gradient = self.back_propagation(input_values, target_values)
                self.update_wb(learning_rate, gradient)
            cost /= len(training_set)
            self.costs.append(cost)
            epoch += 1

        print(f'Epoch: {epoch} Cost: {cost}')

    def predict(self, test_set):
        predictions = []
        for input_values, target_values in test_set:
            output_values = self.forward_pass(input_values)
            if output_values[0] > output_values[1]:
                predictions.append('O')
            else:
                predictions.append('X')
        return predictions
    
    def accuracy(self, test_set):
        predictions = self.predict(test_set)
        correct = 0
        for i in range(len(predictions)):
            print(f'Prediction: {predictions[i]} Label: {test_set[i][1]}')
            if predictions[i] == test_set[i][1]:
                correct += 1
        accuracy = correct / len(test_set) * 100
        return f'Accuracy: {accuracy}%'
    

        
