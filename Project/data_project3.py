import numpy as np
import random
import math
from collections import defaultdict
import csv
import pickle
import time
from datetime import datetime  # Added for timestamp
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Enable interactive plotting
plt.ion()

# Simulation parameters
HIDDEN_NODES = 4
GRID_SIZE = 128
POPULATION_SIZE = 800
GENERATION_LENGTH = 50
SAFE_ZONE_START = 7 * GRID_SIZE // 8
SAFE_ZONE_RADIUS = GRID_SIZE // 8
MUTATION_RATE = 0.001
POPULATION_CAP = 1000
FRAME_DELAY = 0.01  # Delay between frames for visualization

POSSIBLE_SENSORS = [
    'Sir', 'Sfd', 'Sg', 'LPF', 'Plr', 'Pfd', 'Pop',
    'LBf', 'Blr', 'Bfd', 'LMx', 'LMy', 'BDy', 'BDx',
    'BD', 'Lx', 'Ly', 'Age', 'Gen', 'Osc', 'Rnd'
]

POSSIBLE_ACTIONS = [
    'Mfd', 'Mrn', 'Mrv', 'MRL', 'MX', 'MY',
    'LPD', 'Kill', 'OSC', 'SG', 'Res'
]

class Creature:
    def __init__(self, dna=None, position=None, birth_time=0):
        self.position = position if position is not None else np.array([
            random.randint(0, GRID_SIZE-1),
            random.randint(0, GRID_SIZE-1)
        ], dtype=float)
        
        self.dna = dna if dna is not None else self._initialize_dna()
        self.active_sensors = self._get_active_sensors()
        self.active_actions = self._get_active_actions()
        
        # assign a color for visualization
        self.color = (random.random(), random.random(), random.random())
        
        self.sensors = {s: 0 for s in self.active_sensors}
        self.actions = {a: 0 for a in self.active_actions}
        
        self.birth_time = birth_time
        self.last_movement = np.array([1, 0])
        self.oscillator_phase = random.random() * 2 * math.pi
        
        self.long_probe_distance = 5
        self.oscillator_period = 1.0
        self.responsiveness = 0.5
        self.pheromone_strength = 1.0
        
    def _initialize_dna(self):
        dna = []
        for _ in range(15):
            if random.random() < 0.5:
                dna.append((
                    'IH',
                    random.choice(POSSIBLE_SENSORS),
                    random.randint(0, HIDDEN_NODES-1),
                    random.uniform(-1, 1)
                ))
            else:
                dna.append((
                    'HO',
                    random.randint(0, HIDDEN_NODES-1),
                    random.choice(POSSIBLE_ACTIONS),
                    random.uniform(-1, 1)
                ))
        return dna

    def _get_active_sensors(self):
        return list({gene[1] for gene in self.dna if gene[0] == 'IH'})

    def _get_active_actions(self):
        return list({gene[2] for gene in self.dna if gene[0] == 'HO'})

    def mutate(self):
        mutations = 0
        new_dna = []
        
        for gene in self.dna:
            if random.random() < MUTATION_RATE:
                mutated = list(gene)
                mutated[-1] = np.clip(mutated[-1] + random.gauss(0, 0.1), -1, 1)
                if random.random() < 0.1:
                    if mutated[0] == 'IH':
                        mutated[1] = random.choice(POSSIBLE_SENSORS)
                        mutated[2] = random.randint(0, HIDDEN_NODES-1)
                    else:
                        mutated[1] = random.randint(0, HIDDEN_NODES-1)
                        mutated[2] = random.choice(POSSIBLE_ACTIONS)
                new_dna.append(tuple(mutated))
                mutations += 1
            else:
                new_dna.append(gene)
        
        if random.random() < MUTATION_RATE/10:
            if random.random() < 0.5:
                new_dna.append((
                    'IH',
                    random.choice(POSSIBLE_SENSORS),
                    random.randint(0, HIDDEN_NODES-1),
                    random.uniform(-1, 1)
                ))
            else:
                new_dna.append((
                    'HO',
                    random.randint(0, HIDDEN_NODES-1),
                    random.choice(POSSIBLE_ACTIONS),
                    random.uniform(-1, 1)
                ))
            mutations += 1
        
        self.dna = new_dna
        if mutations > 0:
            self.active_sensors = self._get_active_sensors()
            self.active_actions = self._get_active_actions()
        
        return mutations

    def _get_forward_direction(self):
        if np.all(self.last_movement == 0):
            return np.array([1, 0])
        return self.last_movement / np.linalg.norm(self.last_movement)

    def update_sensors(self, grid, current_time, pheromone_grid, creature_grid):
        x, y = map(int, self.position)
        forward_dir = self._get_forward_direction()
        
        if 'Sir' in self.sensors:
            left = pheromone_grid[max(0, x-1), y]
            right = pheromone_grid[min(GRID_SIZE-1, x+1), y]
            self.sensors['Sir'] = np.clip((right - left)/max(1, right+left), -1, 1)

        if 'Sfd' in self.sensors:
            fx, fy = forward_dir
            forward_x, forward_y = int(x+fx), int(y+fy)
            if 0 <= forward_x < GRID_SIZE and 0 <= forward_y < GRID_SIZE:
                forward = pheromone_grid[forward_x, forward_y]
                current = pheromone_grid[x, y]
                self.sensors['Sfd'] = np.clip((forward-current)/max(1, forward+current), -1, 1)

        if 'Sg' in self.sensors:
            self.sensors['Sg'] = np.clip(pheromone_grid[x, y]/10, 0, 1)

        if 'LPF' in self.sensors:
            fx, fy = forward_dir
            scan_x, scan_y = int(x+fx*self.long_probe_distance), int(y+fy*self.long_probe_distance)
            scan_x, scan_y = np.clip([scan_x, scan_y], 0, GRID_SIZE-1)
            self.sensors['LPF'] = np.clip(creature_grid[scan_x, scan_y]/5, 0, 1)

        if 'Plr' in self.sensors:
            left = np.mean(creature_grid[max(0,x-3):x, y])
            right = np.mean(creature_grid[x:min(GRID_SIZE,x+3), y])
            self.sensors['Plr'] = np.clip((right-left)/max(1,right+left), -1, 1)

        if 'Pfd' in self.sensors:
            fx, fy = forward_dir
            forward = np.mean([
                creature_grid[int(x+fx*i), int(y+fy*i)]
                for i in range(1,4)
                if 0 <= int(x+fx*i) < GRID_SIZE and 0 <= int(y+fy*i) < GRID_SIZE
            ])
            current = creature_grid[x, y]
            self.sensors['Pfd'] = np.clip((forward-current)/max(1,forward+current), -1, 1)

        if 'Pop' in self.sensors:
            local = creature_grid[max(0,x-2):min(GRID_SIZE,x+3), max(0,y-2):min(GRID_SIZE,y+3)]
            self.sensors['Pop'] = np.clip(np.sum(local)/25, 0, 1)

        if 'LBf' in self.sensors:
            fx, fy = forward_dir
            for i in range(1, self.long_probe_distance+1):
                check_x, check_y = int(x+fx*i), int(y+fy*i)
                if not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE):
                    self.sensors['LBf'] = 0
                    break
                if creature_grid[check_x, check_y] > 0:
                    self.sensors['LBf'] = 1 - i/self.long_probe_distance
                    break

        if 'Blr' in self.sensors:
            left = any(creature_grid[max(0,x-i),y] > 0 for i in range(1,4))
            right = any(creature_grid[min(GRID_SIZE-1,x+i),y] > 0 for i in range(1,4))
            self.sensors['Blr'] = -1 if left else (1 if right else 0)

        if 'Bfd' in self.sensors:
            fx, fy = forward_dir
            check_x, check_y = int(x+fx), int(y+fy)
            self.sensors['Bfd'] = 1 if (not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE) or creature_grid[check_x, check_y]) > 0 else 0

        if 'LMx' in self.sensors:
            self.sensors['LMx'] = self.last_movement[0]
        if 'LMy' in self.sensors:
            self.sensors['LMy'] = self.last_movement[1]

        if 'BDx' in self.sensors:
            self.sensors['BDx'] = min(x, GRID_SIZE-1-x)/GRID_SIZE
        if 'BDy' in self.sensors:
            self.sensors['BDy'] = min(y, GRID_SIZE-1-y)/GRID_SIZE
        if 'BD' in self.sensors:
            self.sensors['BD'] = min(x, GRID_SIZE-1-x, y, GRID_SIZE-1-y)/GRID_SIZE
        if 'Lx' in self.sensors:
            self.sensors['Lx'] = x/GRID_SIZE
        if 'Ly' in self.sensors:
            self.sensors['Ly'] = y/GRID_SIZE

        if 'Age' in self.sensors:
            self.sensors['Age'] = np.clip((current_time-self.birth_time)/1000, 0, 1)
        if 'Osc' in self.sensors:
            self.oscillator_phase += 0.1/self.oscillator_period
            self.sensors['Osc'] = (math.sin(self.oscillator_phase)+1)/2
        if 'Rnd' in self.sensors:
            self.sensors['Rnd'] = random.random()
        if 'Gen' in self.sensors:
            fx, fy = forward_dir
            check_x, check_y = int(x+fx), int(y+fy)
            if (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE and 
                creature_grid[check_x, check_y] > 0):
                self.sensors['Gen'] = random.uniform(0.7, 1)
            else:
                self.sensors['Gen'] = 0

    def _execute_special_action(self, action):
        if action == 'LPD':
            self.long_probe_distance = random.randint(1, 10)
        elif action == 'OSC':
            self.oscillator_period = random.uniform(0.5, 2.0)
        elif action == 'Res':
            self.responsiveness = random.uniform(0.1, 1.0)
        elif action == 'SG':
            self.pheromone_strength = random.uniform(0.5, 2.0)

    def decide_action(self):
        hidden = np.zeros(HIDDEN_NODES)
        
        for gene in self.dna:
            if gene[0] == 'IH':
                _, sensor, h_idx, weight = gene
                if sensor in self.sensors:
                    hidden[h_idx] += self.sensors[sensor] * weight
        
        hidden = np.tanh(hidden)
        
        action_values = defaultdict(float)
        for gene in self.dna:
            if gene[0] == 'HO':
                _, h_idx, action, weight = gene
                if action in self.actions:
                    action_values[action] += hidden[h_idx] * weight
        
        for action in action_values:
            action_values[action] *= self.responsiveness
        
        if not action_values:
            return None
        
        actions, values = zip(*action_values.items())
        values = np.array(values)
        values = np.nan_to_num(values, nan=0.0)
        
        try:
            exp_values = np.exp(values - np.max(values))
            probabilities = exp_values / np.sum(exp_values)
            selected_action = random.choices(actions, weights=probabilities, k=1)[0]
            self._execute_special_action(selected_action)
            return selected_action
        except:
            return random.choice(list(self.actions.keys())) if self.actions else None

    def move(self, grid, action, pheromone_grid):
        if action is None:
            return False
        
        move_map = {
            'Mfd': self._get_forward_direction(),
            'Mrn': random.choice([np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]),
            'Mrv': -self._get_forward_direction(),
            'MRL': np.array([1 if random.random() > 0.5 else -1, 0]),
            'MX': np.array([1 if random.random() > 0.5 else -1, 0]),
            'MY': np.array([0, 1 if random.random() > 0.5 else -1]),
            'LPD': np.array([0, 0]),
            'Kill': np.array([0, 0]),
            'OSC': np.array([0, 0]),
            'SG': np.array([0, 0]),
            'Res': np.array([0, 0])
        }
        
        movement = move_map.get(action, np.array([0, 0]))
        
        if action == 'Kill':
            fx, fy = self._get_forward_direction()
            target_x, target_y = int(self.position[0] + fx), int(self.position[1] + fy)
            if 0 <= target_x < GRID_SIZE and 0 <= target_y < GRID_SIZE:
                grid[target_x, target_y] = max(0, grid[target_x, target_y] - 1)
            return False
        
        elif action == 'SG':
            x, y = map(int, self.position)
            pheromone_grid[x, y] += self.pheromone_strength
            return False
        
        new_pos = self.position + movement
        new_pos = np.clip(new_pos, 0, GRID_SIZE-1)
        
        if np.array_equal(new_pos, self.position):
            return False
        
        new_pos_int = tuple(map(int, new_pos))
        current_pos_int = tuple(map(int, self.position))
        
        if grid[new_pos_int] == 0:
            grid[current_pos_int] -= 1
            self.position = new_pos
            grid[new_pos_int] += 1
            if action in ['Mfd', 'Mrv', 'MRL', 'MX', 'MY']:
                self.last_movement = movement
            return True
        return False

    def is_in_safe_zone(self):
        """Check if the creature is in one of the quarter-circle safe zones."""
        x, y = self.position
        # bottom-left
        if x**2 + y**2 <= SAFE_ZONE_RADIUS**2:
            return True
        # bottom-right
        if (x - GRID_SIZE)**2 + y**2 <= SAFE_ZONE_RADIUS**2:
            return True
        # top-left
        if x**2 + (y - GRID_SIZE)**2 <= SAFE_ZONE_RADIUS**2:
            return True
        # top-right
        if (x - GRID_SIZE)**2 + (y - GRID_SIZE)**2 <= SAFE_ZONE_RADIUS**2:
            return True
        return False

class Simulation:
    def __init__(self, draw_simulation=True):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.creature_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.pheromone_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.current_time = 0
        self.population = [Creature(birth_time=self.current_time) for _ in range(POPULATION_SIZE)]
        self.generation = 0
        self.survivors_last_gen = 0
        self.draw_simulation = draw_simulation        
        # Metrics tracking
        self.survival_rates = []
        self.genetic_diversities = []
        self.generation_numbers = []
        # Store gene pools
        self.first_generation_gene_pool = None
        self.last_generation_gene_pool = None

        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # timestamp = str(MUTATION_RATE).replace('.', '_')
        self.csv_filename = f'evolution_data {timestamp}.csv'
        self.first_gene_pool_filename = f'first_generation_gene_pool {timestamp}.pkl'
        self.last_gene_pool_filename = f'last_generation_gene_pool {timestamp}.pkl'

        # Visualization setup
        if self.draw_simulation:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, GRID_SIZE)
            self.ax.set_ylim(0, GRID_SIZE)
            self.safe_zone_patch = self.ax.axvspan(SAFE_ZONE_START, GRID_SIZE, color='green', alpha=0.3)
            self.scatter = self.ax.scatter([], [], s=20)
            plt.pause(0.1)

        self._update_grids()

    def _update_grids(self):
        self.grid.fill(0)
        self.creature_grid.fill(0)
        for c in self.population:
            x, y = map(int, c.position)
            self.grid[x, y] += 1
            self.creature_grid[x, y] += 1

    def _calculate_genetic_diversity(self):
        gene_fingerprints = set()
        
        for creature in self.population:
            for gene in creature.dna:
                if gene[0] == 'IH':
                    fingerprint = (gene[0], gene[1], gene[2])
                else:
                    fingerprint = (gene[0], gene[1], gene[2])
                gene_fingerprints.add(fingerprint)
        
        return len(gene_fingerprints)

    def run_frame(self):
        self.current_time += 1
        self.pheromone_grid *= 0.95
        self._update_grids()
        random.shuffle(self.population)
        for c in self.population:
            c.update_sensors(self.grid, self.current_time, self.pheromone_grid, self.creature_grid)
            action = c.decide_action()
            c.move(self.grid, action, self.pheromone_grid)

        if self.draw_simulation:
            positions = np.array([c.position for c in self.population])
            colors = [c.color for c in self.population]
            self.scatter.set_offsets(positions)
            self.scatter.set_color(colors)
            plt.pause(FRAME_DELAY)


    def run_generation(self):
        start_time = time.time()
        
        for _ in range(GENERATION_LENGTH):
            self.run_frame()
        
        # Update survivors based on the new safe zone logic
        survivors = [c for c in self.population if c.is_in_safe_zone()]
        self.survivors_last_gen = len(survivors)
        
        # Calculate metrics
        survival_rate = (len(survivors) / POPULATION_SIZE) * 100
        genetic_diversity = self._calculate_genetic_diversity()
        
        self.survival_rates.append(survival_rate)
        self.genetic_diversities.append(genetic_diversity)
        self.generation_numbers.append(self.generation)
        
        # Save data
        self._save_generation_data()
        self._save_gene_pools()
        
        if len(survivors) == 0:
            print(f"\nTerminating: No survivors in generation {self.generation}")
            return False
        
        # Create next generation
        gene_pool = []
        for survivor in survivors:
            gene_pool.extend(survivor.dna)
        
        new_population = []
        total_mutations = 0
        
        for _ in range(POPULATION_CAP):
            selected_genes = random.choices(gene_pool, k=15)
            
            creature = Creature(
                dna=selected_genes,
                position=np.array([
                    random.randint(0, GRID_SIZE-1),
                    random.randint(0, GRID_SIZE-1)
                ], dtype=float),
                birth_time=self.current_time
            )
            total_mutations += creature.mutate()
            new_population.append(creature)
        
        # Print progress
        gen_time = time.time() - start_time
        if survivors:
            avg_mutations = total_mutations / POPULATION_CAP
            print(f"Gen {self.generation}: Survival={survival_rate:.1f}%, Diversity={genetic_diversity}, Mutations={avg_mutations:.3f}/creature, Time={gen_time:.2f}s")
        
        self.population = new_population
        self.generation += 1
        self._update_grids()
        return True
    def _save_generation_data(self):
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                self.generation,
                self.survival_rates[-1],
                self.genetic_diversities[-1]
            ])
    def _save_gene_pools(self):
        if self.first_generation_gene_pool is None and self.generation == 0:
            self.first_generation_gene_pool = [creature.dna for creature in self.population]
            with open(self.first_gene_pool_filename, 'wb') as f:
                pickle.dump(self.first_generation_gene_pool, f)
        
        self.last_generation_gene_pool = [creature.dna for creature in self.population]
        with open(self.last_gene_pool_filename, 'wb') as f:
            pickle.dump(self.last_generation_gene_pool, f)
    def run(self, n_generations):
        for _ in range(n_generations):
            if not self.run_generation():
                break

if __name__ == '__main__':
    sim = Simulation(draw_simulation=True)
    sim.run(100)  
