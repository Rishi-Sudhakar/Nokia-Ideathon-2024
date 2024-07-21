import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx

class Node:
    def __init__(self, id, x, y, is_tower=False):
        self.id = id
        self.x = x
        self.y = y
        self.is_tower = is_tower
        self.connected_nodes = []
        self.data_queue = deque(maxlen=100)
        self.ai_model = AIRoutingModel()
        self.total_data_processed = 0

    def connect(self, other_node, max_distance):
        distance = np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
        if distance <= max_distance:
            self.connected_nodes.append(other_node)
            other_node.connected_nodes.append(self)
            return True
        return False

    def generate_data(self):
        if not self.is_tower and random.random() < 0.3:
            self.data_queue.append(1)

    def process_data(self):
        processed = 0
        while self.data_queue and processed < 5:
            data = self.data_queue.popleft()
            next_node = self.ai_model.choose_next_node(self, self.connected_nodes)
            if next_node:
                next_node.data_queue.append(data)
                self.total_data_processed += 1
            processed += 1
        return processed

class AIRoutingModel:
    def choose_next_node(self, current_node, connected_nodes):
        if not connected_nodes:
            return None

        # Simple AI model: choose the node with the shortest queue
        return min(connected_nodes, key=lambda node: len(node.data_queue))

class Network:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes = []
        self.max_connection_distance = 200

    def add_node(self, is_tower=False):
        id = len(self.nodes)
        node = Node(id, random.randint(0, self.width), random.randint(0, self.height), is_tower)
        self.nodes.append(node)
        self.connect_node(node)

    def connect_node(self, node):
        for other_node in self.nodes[:-1]:  # Exclude the newly added node
            node.connect(other_node, self.max_connection_distance)

    def simulate(self, ticks):
        data_processed = []
        for _ in range(ticks):
            tick_data = 0
            for node in self.nodes:
                node.generate_data()
                tick_data += node.process_data()
            data_processed.append(tick_data)
        return data_processed

def create_network(num_users, num_towers, width, height):
    network = Network(width, height)
    for _ in range(num_towers):
        network.add_node(is_tower=True)
    for _ in range(num_users):
        network.add_node()
    return network

def run_simulation(num_users, num_towers, width, height, ticks):
    network = create_network(num_users, num_towers, width, height)
    data_processed = network.simulate(ticks)
    return network, data_processed

def plot_network(network):
    G = nx.Graph()
    pos = {}
    colors = []
    sizes = []
    for node in network.nodes:
        G.add_node(node.id)
        pos[node.id] = (node.x, node.y)
        colors.append('red' if node.is_tower else 'blue')
        sizes.append(300 if node.is_tower else 100)
        for connected_node in node.connected_nodes:
            G.add_edge(node.id, connected_node.id)

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, node_size=sizes, with_labels=True)
    plt.title("Network Topology")
    plt.show()

def plot_data_processed(data_processed):
    plt.figure(figsize=(10, 6))
    plt.plot(data_processed)
    plt.title("Data Processed Over Time")
    plt.xlabel("Time Ticks")
    plt.ylabel("Data Processed")
    plt.show()

def main():
    num_users = 50
    num_towers = 5
    width = 1000
    height = 1000
    ticks = 100

    network, data_processed = run_simulation(num_users, num_towers, width, height, ticks)

    plot_network(network)
    plot_data_processed(data_processed)

    print(f"Total data processed: {sum(data_processed)}")
    print(f"Average data processed per tick: {np.mean(data_processed):.2f}")

    # Analysis of node performance
    node_performance = [(node.id, node.total_data_processed, 'Tower' if node.is_tower else 'User')
                        for node in network.nodes]
    node_performance.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 Performing Nodes:")
    for id, data, node_type in node_performance[:5]:
        print(f"Node {id} ({node_type}): {data} data processed")

if __name__ == "__main__":
    main()
