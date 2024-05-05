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