import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation
from collections import defaultdict


# Simulation parameters (MUST MATCH YOUR ORIGINAL SIMULATION)
GRID_SIZE = 128
SAFE_ZONE_START = 7 * GRID_SIZE // 8
HIDDEN_NODES = 10
GENERATION_LENGTH = 50
POPULATION_SIZE = 800
SAFE_ZONE_RADIUS = 10  # Added for safe zone radius

# Parameters you might want to adjust
VISUALIZATION_SPEED = 50  # Milliseconds between frames
CREATURE_SIZE = 10         # Size of creature dots
ALPHA = 0.7               # Transparency of creatures

class Creature:
    def __init__(self, dna=None, position=None):
        self.position = position if position is not None else np.array([
            random.randint(0, GRID_SIZE-1),
            random.randint(0, GRID_SIZE-1)
        ], dtype=float)
        self.dna = dna if dna is not None else []
        self.last_movement = np.array([1, 0])
        self.oscillator_phase = random.random() * 2 * np.pi
        
        # Initialize neural network components
        self.hidden_layer = np.zeros(HIDDEN_NODES)
        self.ih_weights = {}
        self.ho_weights = {}
        self._parse_dna()
        
        # Initialize sensors and actions
        self.sensors = {gene[1]: 0 for gene in self.dna if gene[0] == 'IH'}
        self.actions = {gene[2]: 0 for gene in self.dna if gene[0] == 'HO'}
        
        # Physical properties
        self.long_probe_distance = 5
        self.responsiveness = 0.5
        self.pheromone_strength = 1.0

    def _parse_dna(self):
        """Convert DNA into neural network weights"""
        for gene in self.dna:
            if gene[0] == 'IH':
                _, sensor, h_idx, weight = gene
                self.ih_weights[(sensor, h_idx)] = weight
            elif gene[0] == 'HO':
                _, h_idx, action, weight = gene
                self.ho_weights[(h_idx, action)] = weight

    def _get_forward_direction(self):
        return self.last_movement / np.linalg.norm(self.last_movement) if np.any(self.last_movement) else np.array([1, 0])

    def update_sensors(self, creature_grid, pheromone_grid):
        x, y = map(int, self.position)
        forward_dir = self._get_forward_direction()
        
        # Simplified sensor updates (include all used in original)
        if 'Bfd' in self.sensors:  # Block forward
            fx, fy = forward_dir
            check_x, check_y = int(x+fx), int(y+fy)
            self.sensors['Bfd'] = 1 if (not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE) or creature_grid[check_x, check_y]) > 0 else 0
        
        if 'Sfd' in self.sensors:  # Pheromone forward
            fx, fy = forward_dir
            forward_x, forward_y = int(x+fx), int(y+fy)
            if 0 <= forward_x < GRID_SIZE and 0 <= forward_y < GRID_SIZE:
                self.sensors['Sfd'] = pheromone_grid[forward_x, forward_y] - pheromone_grid[x, y]
        
        if 'BDx' in self.sensors:  # X boundary distance
            self.sensors['BDx'] = min(x, GRID_SIZE-1-x)/GRID_SIZE

    def process_neural_network(self):
        """Run neural network to determine action"""
        # Input to Hidden
        self.hidden_layer[:] = 0
        for (sensor, h_idx), weight in self.ih_weights.items():
            if sensor in self.sensors:
                self.hidden_layer[h_idx] += self.sensors[sensor] * weight
        
        # Hidden activation
        self.hidden_layer = np.tanh(self.hidden_layer)
        
        # Hidden to Output
        action_scores = defaultdict(float)
        for (h_idx, action), weight in self.ho_weights.items():
            if h_idx < HIDDEN_NODES:
                action_scores[action] += self.hidden_layer[h_idx] * weight
        
        return max(action_scores.items(), key=lambda x: x[1])[0] if action_scores else 'Mrn'

    def move(self, creature_grid, pheromone_grid):
        action = self.process_neural_network()
        
        # Movement logic (simplified from original)
        if action == 'Mfd':
            movement = self._get_forward_direction()
        elif action == 'MRL':
            movement = np.array([1, 0])  # Evolved rightward movement
        else:
            movement = random.choice([np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])])
        
        new_pos = np.clip(self.position + movement, 0, GRID_SIZE-1)
        new_pos_int = tuple(map(int, new_pos))
        old_pos_int = tuple(map(int, self.position))
        
        if creature_grid[new_pos_int] == 0:
            creature_grid[old_pos_int] -= 1
            self.position = new_pos
            creature_grid[new_pos_int] += 1
            self.last_movement = movement
            return True
        return False

# Visualization system
class GenerationVisualizer:
    def __init__(self, first_gen_path, last_gen_path, metrics_path):
        # Load saved data
        with open(first_gen_path, 'rb') as f:
            self.first_gen = pickle.load(f)
        with open(last_gen_path, 'rb') as f:
            self.last_gen = pickle.load(f)
        self.metrics = pd.read_csv(metrics_path)
        
        # Initialize simulations
        self.sim1 = self.create_simulation(self.first_gen)
        self.sim2 = self.create_simulation(self.last_gen)
        
        # Set up visualization
        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(2, 2)
        
        # Simulation plots
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[:, 1])  # Metrics plot
        
        # Configure plots
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(0, GRID_SIZE-1)
            ax.set_ylim(0, GRID_SIZE-1)
            self._draw_safe_zones(ax)  # Draw safe zones
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.ax1.set_title("First Generation (Primitive)")
        self.ax2.set_title("Last Generation (Evolved)")
        
        # Metrics plot
        self.ax3.plot(self.metrics['Generation'], self.metrics['SurvivalRate'], 'b-', label='Survival Rate')
        self.ax3.plot(self.metrics['Generation'], self.metrics['GeneticDiversity'], 'r-', label='Genetic Diversity')
        self.ax3.legend()
        self.ax3.set_xlabel('Generation')
        self.ax3.grid(True)
        self.ax3.set_title('Evolution Metrics')
        
        # Create scatter plots
        self.sc1 = self.ax1.scatter([], [], s=CREATURE_SIZE, alpha=ALPHA, c='blue')
        self.sc2 = self.ax2.scatter([], [], s=CREATURE_SIZE, alpha=ALPHA, c='red')

    def _draw_safe_zones(self, ax):
        """Draw full-circle safe zones in the four corners."""
        theta = np.linspace(0, 2 * np.pi, 100)  # Full circle (0 to 2π)

        # Bottom-left corner
        x = SAFE_ZONE_RADIUS * np.cos(theta)
        y = SAFE_ZONE_RADIUS * np.sin(theta)
        ax.fill(x, y, color='lightgreen', alpha=0.3)

        # Bottom-right corner
        x = GRID_SIZE - SAFE_ZONE_RADIUS * np.cos(theta)
        y = SAFE_ZONE_RADIUS * np.sin(theta)
        ax.fill(x, y, color='lightgreen', alpha=0.3)

        # Top-left corner
        x = SAFE_ZONE_RADIUS * np.cos(theta)
        y = GRID_SIZE - SAFE_ZONE_RADIUS * np.sin(theta)
        ax.fill(x, y, color='lightgreen', alpha=0.3)

        # Top-right corner
        x = GRID_SIZE - SAFE_ZONE_RADIUS * np.cos(theta)
        y = GRID_SIZE - SAFE_ZONE_RADIUS * np.sin(theta)
        ax.fill(x, y, color='lightgreen', alpha=0.3)

    def create_simulation(self, gene_pool):
        """Create a simulation instance from a gene pool"""
        sim = {
            'creature_grid': np.zeros((GRID_SIZE, GRID_SIZE)),
            'pheromone_grid': np.zeros((GRID_SIZE, GRID_SIZE)),
            'creatures': [Creature(dna=genes) for genes in gene_pool]
        }
        # Initialize positions
        for c in sim['creatures']:
            x, y = map(int, c.position)
            sim['creature_grid'][x, y] += 1
        return sim

    def update_simulation(self, sim):
        """Update one simulation frame"""
        sim['pheromone_grid'] *= 0.95  # Pheromone decay
        
        # Update creatures
        random.shuffle(sim['creatures'])
        for creature in sim['creatures']:
            creature.update_sensors(sim['creature_grid'], sim['pheromone_grid'])
            creature.move(sim['creature_grid'], sim['pheromone_grid'])
            
            # Deposit pheromones
            x, y = map(int, creature.position)
            sim['pheromone_grid'][x, y] += creature.pheromone_strength
        
        return np.array([c.position for c in sim['creatures']])

    def animate(self, frame):
        """Update animation frame"""
        pos1 = self.update_simulation(self.sim1)
        pos2 = self.update_simulation(self.sim2)
        
        # Update plots
        self.sc1.set_offsets(pos1)
        self.sc2.set_offsets(pos2)
        
        # Update titles
        self.ax1.set_title(f"First Generation - Frame {frame+1}")
        self.ax2.set_title(f"Last Generation - Frame {frame+1}")
        
        return self.sc1, self.sc2

    def run(self):
        """Run the visualization"""
        ani = FuncAnimation(
            self.fig,
            self.animate,
            frames=GENERATION_LENGTH,
            interval=VISUALIZATION_SPEED,
            blit=True,
            repeat = False
        )
        plt.tight_layout()
        plt.show()

# Run the visualization (replace with your actual filenames)
if __name__ == "__main__":
    itr_values = ['20250503_01_55_04']  # Add as many iterations as needed
    for itr in itr_values:
        print(f"Running visualization for itr: {itr}")
        visualizer = GenerationVisualizer(
            first_gen_path=f'Project/Data/first_generation_gene_pool {itr}.pkl',
            last_gen_path=f'Project/Data/last_generation_gene_pool {itr}.pkl',
            metrics_path=f'Project/Data/evolution_data {itr}.csv'
        )
        visualizer.run()