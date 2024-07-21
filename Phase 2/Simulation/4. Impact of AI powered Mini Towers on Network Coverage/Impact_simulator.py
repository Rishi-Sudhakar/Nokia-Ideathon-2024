import random
import numpy as np
import matplotlib.pyplot as plt

class Device:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connected_to = None

class Tower:
    def __init__(self, x, y, range, is_mini=False):
        self.x = x
        self.y = y
        self.range = range
        self.is_mini = is_mini
        self.connected_devices = []

    def connect(self, device):
        if self.is_in_range(device):
            self.connected_devices.append(device)
            device.connected_to = self
            return True
        return False

    def is_in_range(self, device):
        distance = np.sqrt((self.x - device.x)**2 + (self.y - device.y)**2)
        return distance <= self.range

class Network:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.towers = []
        self.devices = []

    def add_tower(self, tower):
        self.towers.append(tower)

    def add_device(self, device):
        self.devices.append(device)

    def connect_devices(self):
        for device in self.devices:
            for tower in self.towers:
                if tower.connect(device):
                    break

    def calculate_coverage(self):
        connected = sum(1 for device in self.devices if device.connected_to is not None)
        return connected / len(self.devices) * 100

    def simulate_ai_routing(self):
        # Simplified AI routing simulation
        for tower in self.towers:
            if len(tower.connected_devices) > 5:  # Arbitrary threshold
                overloaded = tower.connected_devices[5:]
                for device in overloaded:
                    for other_tower in self.towers:
                        if other_tower != tower and len(other_tower.connected_devices) < 5:
                            if other_tower.connect(device):
                                tower.connected_devices.remove(device)
                                break

def run_simulation(num_devices, num_main_towers, num_mini_towers):
    network = Network(1000, 1000)
    
    # Add main towers
    for _ in range(num_main_towers):
        network.add_tower(Tower(random.randint(0, 1000), random.randint(0, 1000), 300))
    
    # Add mini towers
    for _ in range(num_mini_towers):
        network.add_tower(Tower(random.randint(0, 1000), random.randint(0, 1000), 100, is_mini=True))
    
    # Add devices
    for _ in range(num_devices):
        network.add_device(Device(random.randint(0, 1000), random.randint(0, 1000)))
    
    # Connect devices
    network.connect_devices()
    
    # Calculate initial coverage
    initial_coverage = network.calculate_coverage()
    
    # Simulate AI routing
    network.simulate_ai_routing()
    
    # Calculate final coverage
    final_coverage = network.calculate_coverage()
    
    return initial_coverage, final_coverage

# Run multiple simulations
num_simulations = 100
results = []

for _ in range(num_simulations):
    initial, final = run_simulation(num_devices=200, num_main_towers=5, num_mini_towers=20)
    results.append((initial, final))

# Calculate average improvements
avg_initial = sum(r[0] for r in results) / num_simulations
avg_final = sum(r[1] for r in results) / num_simulations

print(f"Average Initial Coverage: {avg_initial:.2f}%")
print(f"Average Final Coverage: {avg_final:.2f}%")
print(f"Average Improvement: {avg_final - avg_initial:.2f}%")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter([r[0] for r in results], [r[1] for r in results], alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line
plt.xlabel('Initial Coverage (%)')
plt.ylabel('Final Coverage (%)')
plt.title('Impact of AI-Powered Mini Towers on Network Coverage')
plt.grid(True)
plt.show()
