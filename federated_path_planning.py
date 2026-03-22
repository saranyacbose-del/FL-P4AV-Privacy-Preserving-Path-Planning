import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import random
import csv
import os
from collections import defaultdict
import math
from typing import List, Tuple, Dict, Set, Optional

# Configuration parameters
CONFIG = {
    "grid_size": 20,
    "num_obstacles": 50,
    "num_vehicles": 5,
    "federated_rounds": 10,
    "local_epochs": 5,
    "learning_rate": 0.01,
    "differential_privacy_noise": 0.1,
    "dynamic_obstacle_prob": 0.05,  # Probability of dynamic obstacle movement
    "output_dir": "results",
    "visualize": True
}

class Environment:
    """2D grid environment with source, destination, and obstacles."""
    
    def __init__(self, size=20, num_obstacles=50):
        self.size = size
        self.grid = np.zeros((size, size))  # 0: empty, 1: obstacle, 2: source, 3: destination
        self.num_obstacles = num_obstacles
        self.initialize_grid()
        
    def initialize_grid(self):
        """Initialize the grid with obstacles, source, and destination."""
        # Reset grid
        self.grid = np.zeros((self.size, self.size))
        
        # Place source (value 2) at the top-left area
        source_x, source_y = random.randint(0, self.size//4), random.randint(0, self.size//4)
        self.source = (source_x, source_y)
        self.grid[source_x, source_y] = 2
        
        # Place destination (value 3) at the bottom-right area
        dest_x, dest_y = random.randint(3*self.size//4, self.size-1), random.randint(3*self.size//4, self.size-1)
        self.destination = (dest_x, dest_y)
        self.grid[dest_x, dest_y] = 3
        
        # Place obstacles (value 1)
        obstacles_placed = 0
        while obstacles_placed < self.num_obstacles:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if self.grid[x, y] == 0:  # Only place on empty cells
                self.grid[x, y] = 1
                obstacles_placed += 1
    
    def is_valid_position(self, x, y):
        """Check if a position is valid (within grid and not an obstacle)."""
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != 1
    
    def get_neighbors(self, x, y):
        """Get valid neighbor positions from current position."""
        possible_neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                             (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        return [(nx, ny) for nx, ny in possible_neighbors if self.is_valid_position(nx, ny)]
    
    def move_dynamic_obstacles(self):
        """Move random obstacles to simulate dynamic environment."""
        obstacle_positions = np.where(self.grid == 1)
        obstacle_coords = list(zip(obstacle_positions[0], obstacle_positions[1]))
        
        for x, y in obstacle_coords:
            if random.random() < CONFIG["dynamic_obstacle_prob"]:
                # Remove current obstacle
                self.grid[x, y] = 0
                
                # Find a new valid position
                for _ in range(10):  # Try up to 10 times
                    new_x = max(0, min(self.size-1, x + random.randint(-1, 1)))
                    new_y = max(0, min(self.size-1, y + random.randint(-1, 1)))
                    
                    # Place at new position if it's empty
                    if self.grid[new_x, new_y] == 0:
                        self.grid[new_x, new_y] = 1
                        break
                else:
                    # If no valid position found, put back in original place
                    self.grid[x, y] = 1
    
    def visualize(self, paths=None, title="Environment Grid"):
        """Visualize the grid with optional paths."""
        plt.figure(figsize=(10, 10))
        
        # Create a colored grid
        colored_grid = np.zeros((self.size, self.size, 3))
        
        # Set colors: white for empty, black for obstacles, red for source, green for destination
        colored_grid[self.grid == 0] = [1, 1, 1]  # Empty (white)
        colored_grid[self.grid == 1] = [0, 0, 0]  # Obstacle (black)
        colored_grid[self.grid == 2] = [1, 0, 0]  # Source (red)
        colored_grid[self.grid == 3] = [0, 1, 0]  # Destination (green)
        
        # Plot the grid
        plt.imshow(colored_grid)
        
        # Plot paths if provided
        if paths:
            for i, path in enumerate(paths):
                if not path:
                    continue
                    
                # Different color for each path
                colors = ['blue', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'brown']
                color = colors[i % len(colors)]
                
                # Extract x and y coordinates
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                
                # Plot the path
                plt.plot(y_coords, x_coords, color=color, linewidth=2, alpha=0.7)
        
        plt.title(title)
        plt.grid(True)
        plt.savefig(f"{CONFIG['output_dir']}/{title.replace(' ', '_')}.png")
        if CONFIG["visualize"]:
            plt.show()
        else:
            plt.close()

    def save_to_csv(self, filename="environment.csv"):
        """Save the grid environment to a CSV file."""
        filepath = os.path.join(CONFIG["output_dir"], filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.grid:
                writer.writerow(row)
        print(f"Environment saved to {filepath}")


class AStar:
    """A* Pathfinding algorithm implementation."""
    
    def __init__(self, environment):
        self.env = environment
    
    def heuristic(self, a, b):
        """Calculate heuristic distance between two points."""
        # Euclidean distance
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    
    def find_path(self, start, goal, weight_map=None):
        """Find a path from start to goal using A* algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            weight_map: Optional weight map for personalized path planning
            
        Returns:
            List of positions representing the path, or None if no path found
        """
        # Initialize the open and closed sets
        open_set = []
        closed_set = set()
        
        # Initial node
        heapq.heappush(open_set, (0, start))
        
        # Dictionary to store the parent of each node
        came_from = {}
        
        # Dictionary to store g-score (cost from start)
        g_score = {start: 0}
        
        # Dictionary to store f-score (estimated total cost)
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            # Get the node with the lowest f-score
            current_f, current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Add the current node to the closed set
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.env.get_neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
                
                # Apply weight map if provided
                if weight_map is not None:
                    # Higher weight means more cost (avoidance)
                    tentative_g += weight_map.get(neighbor, 0)
                
                # If neighbor is not in g_score or we found a better path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update path and scores
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    # Add to open set if not already in
                    if not any(neighbor == n for _, n in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # If we get here, there's no path
        return None


class PrivacyPreservingModel:
    """Privacy-preserving model for personalized path weights."""
    
    def __init__(self, input_dim):
        self.weights = np.random.normal(0, 0.1, input_dim)
        self.bias = np.random.normal(0, 0.1)
    
    def predict(self, x):
        """Generate a path weight based on the input features."""
        return np.dot(x, self.weights) + self.bias
    
    def train(self, x_batch, y_batch, learning_rate):
        """Train the model using gradient descent."""
        for x, y in zip(x_batch, y_batch):
            prediction = self.predict(x)
            error = prediction - y
            
            # Update weights and bias
            self.weights -= learning_rate * error * x
            self.bias -= learning_rate * error
    
    def add_noise(self, scale=0.1):
        """Add differential privacy noise to the model."""
        self.weights += np.random.normal(0, scale, len(self.weights))
        self.bias += np.random.normal(0, scale)


class Vehicle:
    """Autonomous ground vehicle with local data and path planning capabilities."""
    
    def __init__(self, vehicle_id, environment, position=None):
        self.id = vehicle_id
        self.env = environment
        
        # Set position if provided, otherwise use environment's source
        self.position = position if position else environment.source
        
        # Initialize path planning components
        self.path_planner = AStar(environment)
        
        # Feature vector dimensions: [x, y, obstacle_density, distance_to_goal]
        self.model = PrivacyPreservingModel(input_dim=4)
        
        # Local data for training
        self.local_data = self.generate_local_data()
    
    def generate_local_data(self, num_samples=100):
        """Generate local training data based on vehicle's preferences."""
        x_data = []
        y_data = []
        
        # Generate random preferences for this vehicle
        obstacle_avoidance = random.uniform(0.5, 2.0)  # How much to avoid obstacles
        path_length = random.uniform(0.5, 1.5)  # How much to prefer shorter paths
        
        for _ in range(num_samples):
            # Generate random position
            x, y = random.randint(0, self.env.size-1), random.randint(0, self.env.size-1)
            
            # Calculate features
            obstacle_density = self.calculate_obstacle_density(x, y)
            dist_to_goal = math.sqrt((x - self.env.destination[0])**2 + (y - self.env.destination[1])**2)
            
            # Features: [x, y, obstacle_density, distance_to_goal]
            features = np.array([x/self.env.size, y/self.env.size, obstacle_density, dist_to_goal/self.env.size])
            
            # Calculate preferred weight (higher for less desirable paths)
            # This is a simplified example - in real scenarios, this would depend on vehicle's preferences
            weight = obstacle_avoidance * obstacle_density + path_length * (dist_to_goal/self.env.size)
            
            x_data.append(features)
            y_data.append(weight)
        
        return {'x': np.array(x_data), 'y': np.array(y_data)}
    
    def calculate_obstacle_density(self, x, y, radius=2):
        """Calculate obstacle density around a position."""
        count = 0
        total = 0
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.size and 0 <= ny < self.env.size:
                    total += 1
                    if self.env.grid[nx, ny] == 1:  # If obstacle
                        count += 1
        
        return count / total if total > 0 else 0
    
    def train_local_model(self, epochs=5, learning_rate=0.01):
        """Train the local model on local data."""
        x_data = self.local_data['x']
        y_data = self.local_data['y']
        
        # Mini-batch training
        batch_size = min(32, len(x_data))
        for _ in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(x_data))
            x_shuffled = x_data[indices]
            y_shuffled = y_data[indices]
            
            # Train in batches
            for i in range(0, len(x_data), batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.model.train(x_batch, y_batch, learning_rate)
    
    def get_model_weights(self):
        """Get the model weights with differential privacy noise."""
        # Create a copy of the model
        weights_copy = np.copy(self.model.weights)
        bias_copy = self.model.bias
        
        # Add noise for privacy
        noise_scale = CONFIG["differential_privacy_noise"]
        weights_copy += np.random.normal(0, noise_scale, len(weights_copy))
        bias_copy += np.random.normal(0, noise_scale)
        
        return {'weights': weights_copy, 'bias': bias_copy}
    
    def update_model(self, global_weights, global_bias):
        """Update local model with global aggregated weights."""
        # Blend local and global weights (personalization)
        blend_factor = 0.8  # Higher value means more personalization
        self.model.weights = blend_factor * self.model.weights + (1 - blend_factor) * global_weights
        self.model.bias = blend_factor * self.model.bias + (1 - blend_factor) * global_bias
    
    def calculate_path_weights(self):
        """Calculate path weights for the grid based on the trained model."""
        weight_map = {}
        
        for x in range(self.env.size):
            for y in range(self.env.size):
                # Skip obstacles, source, and destination
                if self.env.grid[x, y] in [1, 2, 3]:
                    continue
                
                # Calculate features
                obstacle_density = self.calculate_obstacle_density(x, y)
                dist_to_goal = math.sqrt((x - self.env.destination[0])**2 + (y - self.env.destination[1])**2)
                
                # Create feature vector
                features = np.array([x/self.env.size, y/self.env.size, obstacle_density, dist_to_goal/self.env.size])
                
                # Predict weight using the model
                weight = max(0, self.model.predict(features))
                
                weight_map[(x, y)] = weight
        
        return weight_map
    
    def find_path(self):
        """Find a path from current position to destination using personalized weights."""
        # Calculate path weights
        weight_map = self.calculate_path_weights()
        
        # Use A* with the weight map
        path = self.path_planner.find_path(self.position, self.env.destination, weight_map)
        
        return path
    
    def update_position(self, new_position):
        """Update the vehicle's position."""
        if self.env.is_valid_position(*new_position):
            self.position = new_position
            return True
        return False
    
    def export_data(self, filename=None):
        """Export vehicle data to CSV."""
        if filename is None:
            filename = f"vehicle_{self.id}_data.csv"
        
        filepath = os.path.join(CONFIG["output_dir"], filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write headers
            writer.writerow(['x', 'y', 'obstacle_density', 'distance_to_goal', 'weight'])
            
            # Write data
            for i in range(len(self.local_data['x'])):
                row = list(self.local_data['x'][i]) + [self.local_data['y'][i]]
                writer.writerow(row)
        
        print(f"Vehicle {self.id} data exported to {filepath}")


class FederatedLearning:
    """Federated Learning coordinator for privacy-preserving path planning."""
    
    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.rounds = CONFIG["federated_rounds"]
        self.local_epochs = CONFIG["local_epochs"]
        self.learning_rate = CONFIG["learning_rate"]
        
        # Global model weights (for aggregation)
        self.global_weights = None
        self.global_bias = None
    
    def run_federated_learning(self):
        """Run the federated learning process."""
        print("Starting federated learning process...")
        
        for round_num in range(1, self.rounds + 1):
            print(f"\nFederated Learning Round {round_num}/{self.rounds}")
            
            # 1. Local training on each vehicle
            for vehicle in self.vehicles:
                vehicle.train_local_model(epochs=self.local_epochs, learning_rate=self.learning_rate)
            
            # 2. Collect model updates with differential privacy
            vehicle_weights = []
            vehicle_biases = []
            
            for vehicle in self.vehicles:
                model_params = vehicle.get_model_weights()
                vehicle_weights.append(model_params['weights'])
                vehicle_biases.append(model_params['bias'])
            
            # 3. Aggregate model updates (federated averaging)
            self.global_weights = np.mean(vehicle_weights, axis=0)
            self.global_bias = np.mean(vehicle_biases)
            
            # 4. Distribute global model back to vehicles
            for vehicle in self.vehicles:
                vehicle.update_model(self.global_weights, self.global_bias)
            
            print(f"Round {round_num} completed.")
        
        print("\nFederated learning process completed.")
    
    def evaluate_paths(self):
        """Evaluate and visualize paths for all vehicles."""
        paths = []
        
        for vehicle in self.vehicles:
            path = vehicle.find_path()
            paths.append(path)
            
            # Print path info
            print(f"Vehicle {vehicle.id} path length: {len(path) if path else 'No path found'}")
        
        # Visualize all paths
        self.vehicles[0].env.visualize(paths, f"Federated Path Planning - All Vehicles")
        
        return paths
    
    def export_results(self):
        """Export federated learning results."""
        # Export global model weights
        filepath = os.path.join(CONFIG["output_dir"], "global_model.csv")
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['parameter', 'value'])
            for i, w in enumerate(self.global_weights):
                writer.writerow([f'weight_{i}', w])
            writer.writerow(['bias', self.global_bias])
        
        print(f"Global model exported to {filepath}")


class Simulation:
    """Simulation for the federated learning path planning system."""
    
    def __init__(self):
        # Create output directory if it doesn't exist
        if not os.path.exists(CONFIG["output_dir"]):
            os.makedirs(CONFIG["output_dir"])
        
        # Initialize environment
        self.env = Environment(size=CONFIG["grid_size"], num_obstacles=CONFIG["num_obstacles"])
        
        # Initialize vehicles
        self.vehicles = []
        for i in range(CONFIG["num_vehicles"]):
            # Create vehicles at random positions near the source
            x = self.env.source[0] + random.randint(-2, 2)
            y = self.env.source[1] + random.randint(-2, 2)
            
            # Ensure position is valid
            x = max(0, min(self.env.size-1, x))
            y = max(0, min(self.env.size-1, y))
            
            vehicle = Vehicle(i, self.env, position=(x, y))
            self.vehicles.append(vehicle)
        
        # Initialize federated learning
        self.federated_learning = FederatedLearning(self.vehicles)
    
    def run(self):
        """Run the simulation."""
        print("Starting simulation...")
        
        # Visualize initial environment
        self.env.visualize(title="Initial Environment")
        
        # Export environment to CSV
        self.env.save_to_csv()
        
        # Run federated learning
        self.federated_learning.run_federated_learning()
        
        # Evaluate paths
        paths = self.federated_learning.evaluate_paths()
        
        # Export results
        self.federated_learning.export_results()
        
        # Simulate dynamic obstacles
        print("\nSimulating dynamic obstacles...")
        for i in range(5):  # 5 timesteps
            # Move obstacles
            self.env.move_dynamic_obstacles()
            
            # Re-plan paths
            dynamic_paths = []
            for vehicle in self.vehicles:
                path = vehicle.find_path()
                dynamic_paths.append(path)
            
            # Visualize dynamic environment
            self.env.visualize(dynamic_paths, f"Dynamic Environment - Timestep {i+1}")
            
            # Export dynamic environment
            self.env.save_to_csv(f"dynamic_environment_t{i+1}.csv")
        
        print("\nSimulation completed.")


def main():
    """Main function to run the simulation."""
    print("Federated Learning-Enabled Privacy-Preserving Path Planning for Collaborative AGVs")
    print("=" * 80)
    
    # Start simulation
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()