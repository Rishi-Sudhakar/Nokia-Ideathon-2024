import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
import pandas as pd
from collections import deque

class Device:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connected_to = None
        self.data_queue = deque(maxlen=50)
        self.total_data_sent = 0
        self.total_latency = 0
        self.packets_sent = 0

    def generate_data(self):
        if random.random() < 0.3:
            self.data_queue.append(1)

    def send_data(self):
        if self.connected_to and self.data_queue:
            data = self.data_queue.popleft()
            self.total_data_sent += data
            self.packets_sent += 1
            latency = self.connected_to.receive_data(data)
            self.total_latency += latency
            return data
        return 0

    def get_average_latency(self):
        return self.total_latency / self.packets_sent if self.packets_sent > 0 else 0

class BaseTower:
    def __init__(self, x, y, range):
        self.x = x
        self.y = y
        self.range = range
        self.connected_devices = []
        self.data_queue = deque(maxlen=100)
        self.total_data_processed = 0

    def connect(self, device):
        if self.is_in_range(device):
            self.connected_devices.append(device)
            device.connected_to = self
            return True
        return False

    def is_in_range(self, obj):
        distance = np.sqrt((self.x - obj.x)**2 + (self.y - obj.y)**2)
        return distance <= self.range

    def receive_data(self, data):
        self.data_queue.append(data)
        self.total_data_processed += data
        return random.uniform(0.1, 0.5)  # Simulated latency

    def process_data(self):
        processed = 0
        while self.data_queue and processed < 10:
            self.data_queue.popleft()
            processed += 1
        return processed

class TraditionalTower(BaseTower):
    def __init__(self, x, y):
        super().__init__(x, y, range=500)

class MiniTower(BaseTower):
    def __init__(self, x, y):
        super().__init__(x, y, range=200)
        self.connected_towers = []
        self.ai_model = AIModel()

    def connect_tower(self, tower):
        if self.is_in_range(tower):
            self.connected_towers.append(tower)
            return True
        return False

    def route_data(self):
        return self.ai_model.optimize_routing(self.data_queue, self.connected_towers)

class AIModel:
    def optimize_routing(self, data_queue, connected_towers):
        if not connected_towers:
            return 0

        tower_loads = [len(tower.data_queue) for tower in connected_towers]
        total_load = sum(tower_loads)
        
        if total_load == 0:
            return 0

        inverse_loads = [1 / (load + 1) for load in tower_loads]
        total_inverse = sum(inverse_loads)
        distribution = [inv / total_inverse for inv in inverse_loads]

        data_sent = 0
        while data_queue and data_sent < 10:
            data = data_queue.popleft()
            tower_index = random.choices(range(len(connected_towers)), weights=distribution)[0]
            connected_towers[tower_index].data_queue.append(data)
            data_sent += 1

        return data_sent

class Network:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.traditional_towers = []
        self.mini_towers = []
        self.devices = []

    def add_traditional_tower(self, tower):
        self.traditional_towers.append(tower)

    def add_mini_tower(self, tower):
        self.mini_towers.append(tower)
        for other_tower in self.mini_towers[:-1]:
            tower.connect_tower(other_tower)
            other_tower.connect_tower(tower)

    def add_device(self, device):
        self.devices.append(device)

    def connect_devices(self):
        for device in self.devices:
            all_towers = self.traditional_towers + self.mini_towers
            random.shuffle(all_towers)
            for tower in all_towers:
                if tower.connect(device):
                    break

    def simulate_network(self, ticks):
        coverage_over_time = []
        data_processed_traditional = []
        data_processed_mini = []

        for _ in range(ticks):
            for device in self.devices:
                device.generate_data()
                device.send_data()

            traditional_processed = sum(tower.process_data() for tower in self.traditional_towers)
            data_processed_traditional.append(traditional_processed)

            mini_processed = sum(tower.route_data() for tower in self.mini_towers)
            data_processed_mini.append(mini_processed)

            connected = sum(1 for device in self.devices if device.connected_to is not None)
            coverage_over_time.append(connected / len(self.devices) * 100)

        return coverage_over_time, data_processed_traditional, data_processed_mini

def run_simulation(num_devices, num_traditional_towers, num_mini_towers, simulation_ticks):
    network = Network(1000, 1000)
    
    for _ in range(num_traditional_towers):
        network.add_traditional_tower(TraditionalTower(random.randint(0, 1000), random.randint(0, 1000)))
    
    for _ in range(num_mini_towers):
        network.add_mini_tower(MiniTower(random.randint(0, 1000), random.randint(0, 1000)))
    
    for _ in range(num_devices):
        network.add_device(Device(random.randint(0, 1000), random.randint(0, 1000)))
    
    network.connect_devices()
    
    coverage, data_traditional, data_mini = network.simulate_network(simulation_ticks)
    
    avg_latency = sum(device.get_average_latency() for device in network.devices) / num_devices
    
    return coverage, data_traditional, data_mini, avg_latency

def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # Coverage over time
    plt.subplot(2, 2, 1)
    for i, (coverage, _, _, _) in enumerate(results):
        plt.plot(range(len(coverage)), coverage, label=f'Simulation {i+1}')
    plt.xlabel('Time (ticks)')
    plt.ylabel('Network Coverage (%)')
    plt.title('Network Coverage Over Time')
    plt.legend()

    # Data processed comparison
    plt.subplot(2, 2, 2)
    traditional_data = [sum(r[1]) for r in results]
    mini_data = [sum(r[2]) for r in results]
    x = range(len(results))
    width = 0.35
    plt.bar([i - width/2 for i in x], traditional_data, width, label='Traditional Towers')
    plt.bar([i + width/2 for i in x], mini_data, width, label='Mini Towers')
    plt.xlabel('Simulation')
    plt.ylabel('Total Data Processed')
    plt.title('Data Processing Comparison')
    plt.legend()

    # Average latency
    plt.subplot(2, 2, 3)
    latencies = [r[3] for r in results]
    plt.bar(range(len(latencies)), latencies)
    plt.xlabel('Simulation')
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Network Latency')

    # Network efficiency
    plt.subplot(2, 2, 4)
    traditional_efficiency = [sum(r[1]) / len(r[1]) for r in results]
    mini_efficiency = [sum(r[2]) / len(r[2]) for r in results]
    plt.scatter(traditional_efficiency, mini_efficiency)
    plt.plot([0, max(traditional_efficiency + mini_efficiency)], [0, max(traditional_efficiency + mini_efficiency)], 'r--')
    plt.xlabel('Traditional Tower Efficiency')
    plt.ylabel('Mini Tower Efficiency')
    plt.title('Network Efficiency Comparison')

    plt.tight_layout()
    plt.show()

def run_parameter_sweep():
    device_range = [100, 500, 1000]
    traditional_tower_range = [5, 10, 20]
    mini_tower_range = [20, 50, 100]
    
    results = []
    
    for devices in device_range:
        for trad_towers in traditional_tower_range:
            for mini_towers in mini_tower_range:
                coverage, data_trad, data_mini, latency = run_simulation(devices, trad_towers, mini_towers, 100)
                results.append({
                    'devices': devices,
                    'traditional_towers': trad_towers,
                    'mini_towers': mini_towers,
                    'final_coverage': coverage[-1],
                    'total_data_traditional': sum(data_trad),
                    'total_data_mini': sum(data_mini),
                    'average_latency': latency
                })
    
    return results

def plot_parameter_sweep(results):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(20, 15))
    
    # Network Coverage by Tower Configuration
    plt.subplot(2, 2, 1)
    for devices in df['devices'].unique():
        subset = df[df['devices'] == devices]
        plt.scatter(subset['traditional_towers'], subset['mini_towers'], 
                    s=subset['final_coverage'], alpha=0.6, 
                    label=f'{devices} devices')
    plt.xlabel('Traditional Towers')
    plt.ylabel('Mini Towers')
    plt.title('Network Coverage by Tower Configuration')
    plt.legend()
    
    # Data Processed vs Number of Devices
    plt.subplot(2, 2, 2)
    df_grouped = df.groupby('devices')[['total_data_traditional', 'total_data_mini']].mean()
    df_grouped.plot(kind='bar', ax=plt.gca())
    plt.title('Average Data Processed vs Number of Devices')
    plt.xlabel('Number of Devices')
    plt.ylabel('Average Data Processed')
    
    # Average Latency Heatmap
    plt.subplot(2, 2, 3)
    latency_data = df.groupby(['traditional_towers', 'mini_towers'])['average_latency'].mean().unstack()
    sns.heatmap(latency_data, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Average Latency by Tower Configuration')
    plt.xlabel('Mini Towers')
    plt.ylabel('Traditional Towers')
    
    # Coverage Distribution by Number of Devices
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='devices', y='final_coverage')
    plt.title('Coverage Distribution by Number of Devices')
    plt.xlabel('Number of Devices')
    plt.ylabel('Final Coverage (%)')
    
    plt.tight_layout()
    plt.show()

# Run simulations
num_simulations = 5
simulation_results = []

for i in range(num_simulations):
    print(f"Running simulation {i+1}/{num_simulations}")
    result = run_simulation(num_devices=500, num_traditional_towers=10, num_mini_towers=50, simulation_ticks=100)
    simulation_results.append(result)

# Plot results
plot_results(simulation_results)

# Run parameter sweep
print("Running parameter sweep...")
sweep_results = run_parameter_sweep()
plot_parameter_sweep(sweep_results)