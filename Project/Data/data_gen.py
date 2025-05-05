import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation

# Simulation parameters (MUST MATCH YOUR ORIGINAL SIMULATION)
GRID_SIZE = 128
SAFE_ZONE_START = 7 * GRID_SIZE // 8
HIDDEN_NODES = 10
GENERATION_LENGTH = 50
POPULATION_SIZE = 800
SAFE_ZONE_RADIUS = 10  # Added for safe zone radius

# Parameters you might want to adjust
VISUALIZATION_SPEED = 200  # Milliseconds between frames
CREATURE_SIZE = 10         # Size of creature dots
ALPHA = 0.7               # Transparency of creatures

class GenerationVisualizer:
    def __init__(self, num_creatures=POPULATION_SIZE):
        """Initialize the visualization with two groups of creatures."""
        self.num_creatures = num_creatures
        self.creatures_lr = self._initialize_creatures_lr()  # Left-right moving creatures
        self.creatures_ud = self._initialize_creatures_ud()  # Up-down moving creatures
        
        # Set up visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, GRID_SIZE - 1)
        self.ax.set_ylim(0, GRID_SIZE - 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Creature Movement Simulation")
        
        # Create scatter plots for both groups
        self.scatter_lr = self.ax.scatter(
            [c['position'][0] for c in self.creatures_lr],
            [c['position'][1] for c in self.creatures_lr],
            s=CREATURE_SIZE,
            alpha=ALPHA,
            c='blue',
         )
        self.scatter_ud = self.ax.scatter(
            [c['position'][0] for c in self.creatures_ud],
            [c['position'][1] for c in self.creatures_ud],
            s=CREATURE_SIZE,
            alpha=ALPHA,
            c='blue',
         )
        self.ax.legend()

    def _initialize_creatures_lr(self):
        """Initialize creatures that move left or right."""
        return [
            {
                'position': np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]),
                'direction': random.choice([-1, 1]),  # -1 for left, 1 for right
                'vertical_direction': None  # Will be set when reaching the edge
            }
            for _ in range(int(self.num_creatures * 0.8))  # 80% of the creatures
        ]

    def _initialize_creatures_ud(self):
        """Initialize creatures that move up or down."""
        return [
            {
                'position': np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]),
                'direction': random.choice([-1, 1])  # -1 for up, 1 for down
            }
            for _ in range(int(self.num_creatures * 0.1))  # 10% of the creatures
        ]

    def _update_creatures_lr(self):
        """Update the positions of left-right moving creatures."""
        for creature in self.creatures_lr:
            if creature['vertical_direction'] is None:  # Moving left or right
                creature['position'][0] += creature['direction']
                # Check if the creature has reached the side edges
                if creature['position'][0] <= 0 or creature['position'][0] >= GRID_SIZE - 1:
                    creature['vertical_direction'] = random.choice([-1, 1])  # Choose up or down
            else:  # Moving up or down after reaching the edge
                creature['position'][1] = np.clip(creature['position'][1] + creature['vertical_direction'], 0, GRID_SIZE - 1)

    def _update_creatures_ud(self):
        """Update the positions of up-down moving creatures."""
        for creature in self.creatures_ud:
            creature['position'][1] = np.clip(creature['position'][1] + creature['direction'], 0, GRID_SIZE - 1)

    def animate(self, frame):
        """Update animation frame."""
        self._update_creatures_lr()
        self._update_creatures_ud()
        self.scatter_lr.set_offsets([c['position'] for c in self.creatures_lr])
        self.scatter_ud.set_offsets([c['position'] for c in self.creatures_ud])
        return self.scatter_lr, self.scatter_ud

    def run(self):
        """Run the visualization."""
        ani = FuncAnimation(
            self.fig,
            self.animate,
            frames=GENERATION_LENGTH,
            interval=VISUALIZATION_SPEED,
            blit=True,
            repeat=False
        )
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    visualizer = GenerationVisualizer(num_creatures=POPULATION_SIZE)  # Use POPULATION_SIZE constant
    visualizer.run()