import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class Node:
    def __init__(self, id, x, y, node_type):
        self.id = id
        self.x = x
        self.y = y
        self.node_type = node_type
        self.connected_nodes = []
        self.data_queue = deque(maxlen=100)
        self.total_data_processed = 0
        self.total_latency = 0
        self.packets_sent = 0

    def connect(self, other_node, max_distance):
        distance = np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
        if distance <= max_distance:
            self.connected_nodes.append(other_node)
            return True
        return False

    def generate_data(self):
        if self.node_type == "user" and random.random() < 0.3:
            self.data_queue.append((1, random.choice(["google", "facebook"])))

    def process_data(self, ai_model):
        processed = 0
        while self.data_queue and processed < 5:
            data, destination = self.data_queue.popleft()
            next_node = ai_model.choose_next_node(self, self.connected_nodes, destination)
            if next_node:
                latency = self.calculate_latency(next_node)
                next_node.data_queue.append((data, destination))
                self.total_data_processed += 1
                self.total_latency += latency
                self.packets_sent += 1
            processed += 1
        return processed

    def calculate_latency(self, next_node):
        distance = np.sqrt((self.x - next_node.x)**2 + (self.y - next_node.y)**2)
        base_latency = distance / 100  # Simplified latency calculation
        return base_latency + random.uniform(0, 0.5)  # Add some randomness

    def get_average_latency(self):
        return self.total_latency / self.packets_sent if self.packets_sent > 0 else 0

class AIRoutingModel:
    def choose_next_node(self, current_node, connected_nodes, destination):
        if not connected_nodes:
            return None

        # Priority: Server > Traditional Tower > Mini Tower > User
        type_priority = {"server": 0, "traditional_tower": 1, "mini_tower": 2, "user": 3}
        
        # Sort connected nodes by type priority and queue length
        sorted_nodes = sorted(connected_nodes, 
                              key=lambda node: (type_priority[node.node_type], 
                                                len(node.data_queue)))
        
        for node in sorted_nodes:
            if node.node_type == "server" and node.id.startswith(destination):
                return node  # Direct connection to the destination server
        
        return sorted_nodes[0]  # Return the best node based on type and queue length

class Network:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes = []
        self.max_connection_distance = {
            "traditional_tower": 500,
            "mini_tower": 200,
            "user": 100,
            "server": 1000
        }

    def add_node(self, node_type, x=None, y=None):
        id = f"{node_type}_{len([n for n in self.nodes if n.node_type == node_type])}"
        x = x if x is not None else random.randint(0, self.width)
        y = y if y is not None else random.randint(0, self.height)
        node = Node(id, x, y, node_type)
        self.nodes.append(node)
        self.connect_node(node)

    def connect_node(self, node):
        for other_node in self.nodes[:-1]:  # Exclude the newly added node
            max_distance = max(self.max_connection_distance[node.node_type],
                               self.max_connection_distance[other_node.node_type])
            node.connect(other_node, max_distance)

    def simulate(self, ticks):
        ai_model = AIRoutingModel()
        data_processed = []
        for _ in range(ticks):
            tick_data = 0
            for node in self.nodes:
                node.generate_data()
                tick_data += node.process_data(ai_model)
            data_processed.append(tick_data)
        return data_processed

def create_network(num_users, num_mini_towers, num_traditional_towers, width, height):
    network = Network(width, height)
    
    # Add servers (Google and Facebook) at fixed positions
    network.add_node("server", x=width//4, y=height//2)  # Google
    network.add_node("server", x=3*width//4, y=height//2)  # Facebook
    
    for _ in range(num_traditional_towers):
        network.add_node("traditional_tower")
    for _ in range(num_mini_towers):
        network.add_node("mini_tower")
    for _ in range(num_users):
        network.add_node("user")
    
    return network

def run_simulation(num_users, num_mini_towers, num_traditional_towers, width, height, ticks):
    network = create_network(num_users, num_mini_towers, num_traditional_towers, width, height)
    data_processed = network.simulate(ticks)
    return network, data_processed

def plot_network(network):
    G = nx.Graph()
    pos = {}
    colors = []
    sizes = []
    labels = {}
    
    color_map = {
        "user": "#FFA500",  # Orange
        "mini_tower": "#00CED1",  # Dark Turquoise
        "traditional_tower": "#FF1493",  # Deep Pink
        "server": "#32CD32"  # Lime Green
    }
    
    size_map = {
        "user": 100,
        "mini_tower": 300,
        "traditional_tower": 500,
        "server": 800
    }

    for node in network.nodes:
        G.add_node(node.id)
        pos[node.id] = (node.x, node.y)
        colors.append(color_map[node.node_type])
        sizes.append(size_map[node.node_type])
        labels[node.id] = node.id if node.node_type != "user" else ""
        for connected_node in node.connected_nodes:
            G.add_edge(node.id, connected_node.id)

    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, node_color=colors, node_size=sizes, with_labels=False, alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=node_type.replace('_', ' ').title(),
                  markerfacecolor=color, markersize=10) for node_type, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Network Topology")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_data_processed(data_processed):
    plt.figure(figsize=(12, 6))
    plt.plot(data_processed)
    plt.title("Data Processed Over Time")
    plt.xlabel("Time Ticks")
    plt.ylabel("Data Packets Processed")
    plt.grid(True)
    plt.show()

def plot_node_performance(network):
    node_types = ["user", "mini_tower", "traditional_tower", "server"]
    performances = {node_type: [] for node_type in node_types}
    latencies = {node_type: [] for node_type in node_types}

    for node in network.nodes:
        performances[node.node_type].append(node.total_data_processed)
        latencies[node.node_type].append(node.get_average_latency())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Data Processed Box Plot
    ax1.boxplot([performances[nt] for nt in node_types])
    ax1.set_xticklabels([nt.replace('_', ' ').title() for nt in node_types])
    ax1.set_ylabel("Total Data Processed")
    ax1.set_title("Data Processing Performance by Node Type")

    # Latency Box Plot
    ax2.boxplot([latencies[nt] for nt in node_types])
    ax2.set_xticklabels([nt.replace('_', ' ').title() for nt in node_types])
    ax2.set_ylabel("Average Latency (ms)")
    ax2.set_title("Latency Performance by Node Type")

    plt.tight_layout()
    plt.show()

def main():
    num_users = 100
    num_mini_towers = 80
    #(Considering around 80% of them use the mini towers)
    num_traditional_towers = 3
    width = 1000
    height = 1000
    ticks = 200

    network, data_processed = run_simulation(num_users, num_mini_towers, num_traditional_towers, width, height, ticks)

    plot_network(network)
    plot_data_processed(data_processed)
    plot_node_performance(network)

    print(f"Total data processed: {sum(data_processed)}")
    print(f"Average data processed per tick: {np.mean(data_processed):.2f}")

    # Analysis of node performance
    node_performance = [(node.id, node.total_data_processed, node.get_average_latency(), node.node_type)
                        for node in network.nodes]
    node_performance.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 Performing Nodes:")
    for id, data, latency, node_type in node_performance[:5]:
        print(f"{id} ({node_type}): {data} data processed, {latency:.2f}ms avg latency")

if __name__ == "__main__":
    main()
