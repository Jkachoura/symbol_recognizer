import random
class Link:

    def __init__(self, from_node, to_node):
        self.weight = random.uniform(-1, 1)
        self.from_node = from_node
        self.to_node = to_node