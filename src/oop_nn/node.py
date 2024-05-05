import math as m
import random

class Node:

    def __init__(self):
        self.outgoing_links = []
        self.incoming_links = []
        self.value = 0.0
        self.bias = random.uniform(-1, 1)

    def calculate_sum(self):
        input_value = 0
        for link in self.incoming_links:
            input_value += link.from_node.value * link.weight
        return input_value + self.bias
    
    def softmax(self):
        #TODO: Implement softmax function
        return 0
    
    def sigmoid(self):
        return 1 / (1 + m.exp(-self.value))
   
    def calculate_value(self):
        self.value = self.calculate_sum()
        # self.value = self.softmax()
        self.value = self.sigmoid()
        return self.value