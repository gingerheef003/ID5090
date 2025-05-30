import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from collections import defaultdict
import datetime  # Add this import at the top of the file

plt.ion()

# Simulation parameters
HIDDEN_NODES = 7
FRAME_DELAY = 0.1
GRID_SIZE = 128
POPULATION_SIZE = 800
GENERATION_LENGTH = 150
SAFE_ZONE_START = 4* GRID_SIZE // 8
MUTATION_RATE = 0.001
POPULATION_CAP = 800
GENE_LENGTH = 15
MAX_GENERATIONS = 200
DRAW_SIM = False

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
        
        # Initialize sensors and actions dictionaries
        self.sensors = {s: 0 for s in self.active_sensors}
        self.actions = {a: 0 for a in self.active_actions}
        
        self.color = self._generate_color()
        self.birth_time = birth_time
        self.last_movement = np.array([1, 0])
        self.oscillator_phase = random.random() * 2 * math.pi
        self.genetic_signature = random.random()
        
        self.long_probe_distance = 5
        self.oscillator_period = 1.0
        self.responsiveness = 0.5
        self.pheromone_strength = 1.0
        
    def _initialize_dna(self):
        dna = []
        for _ in range(GENE_LENGTH):
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

    def _generate_color(self):
        # Create DNA fingerprint ignoring weight values
        dna_fingerprint = tuple(sorted(
            (g[0], g[1], g[2]) if g[0] == 'IH' else (g[0], g[1], g[2])
            for g in self.dna
        ))
        hue = hash(dna_fingerprint) % 360 / 360
        return plt.cm.hsv(hue)

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
            self.color = self._generate_color()
        
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
            self.sensors['Bfd'] = 1 if (not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE) or creature_grid[check_x, check_y] )> 0 else 0

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

class Simulation:
    def __init__(self, generations_to_track=50, draw_simulation=DRAW_SIM):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.creature_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.pheromone_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        self.current_time = 0
        self.population = [Creature(birth_time=self.current_time) for _ in range(POPULATION_SIZE)]
        self.generation = 0
        self.survivors_last_gen = 0
        self.current_frame = 0
        
        # Metrics tracking
        self.generations_to_track = generations_to_track
        self.survival_rates = []
        self.genetic_diversities = []
        self.generation_numbers = []
        
        # Visualization flag
        self.draw_simulation = draw_simulation
        
        if self.draw_simulation:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_xlim(0, GRID_SIZE-1)
            self.ax.set_ylim(0, GRID_SIZE-1)
            self.ax.set_title(f"Evolution Simulation (Hidden Nodes: {HIDDEN_NODES})")
            self.safe_zone_patch = self.ax.axvspan(SAFE_ZONE_START, GRID_SIZE-1, facecolor='#90EE90', alpha=0.3)
            self.ax.plot([0, GRID_SIZE-1, GRID_SIZE-1, 0, 0], 
                        [0, 0, GRID_SIZE-1, GRID_SIZE-1, 0], 
                        'k-', linewidth=2)
            
            self.creature_plot = self.ax.scatter([], [], s=100, alpha=0.9, edgecolor='black')
            self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,
                                        bbox=dict(facecolor='white', alpha=0.8))
        
        self._update_grids()

    def _update_grids(self):
        self.grid.fill(0)
        self.creature_grid.fill(0)
        for creature in self.population:
            x, y = map(int, creature.position)
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self.grid[x, y] += 1
                self.creature_grid[x, y] += 1

    def _update_safe_zone_visualization(self):
        """Update the color of the safe zone based on the current generation."""
        if not self.draw_simulation:
            return
        
        if self.generation < MAX_GENERATIONS // 2:
            # Right part is the safe zone
            self.safe_zone_patch.remove()
            self.safe_zone_patch = self.ax.axvspan(SAFE_ZONE_START, GRID_SIZE-1, facecolor='#90EE90', alpha=0.3)
        else:
            # Left part is the safe zone
            self.safe_zone_patch.remove()
            self.safe_zone_patch = self.ax.axvspan(0, SAFE_ZONE_START, facecolor='#ADD8E6', alpha=0.3)

    def _calculate_genetic_diversity(self):
        # Count unique gene combinations (ignoring weights)
        gene_fingerprints = set()
        
        for creature in self.population:
            for gene in creature.dna:
                # Create fingerprint without the weight
                if gene[0] == 'IH':
                    fingerprint = (gene[0], gene[1], gene[2])
                else:
                    fingerprint = (gene[0], gene[1], gene[2])
                gene_fingerprints.add(fingerprint)
        
        return len(gene_fingerprints)

    def _update_visualization(self):
        if not self.draw_simulation:
            return
        
        positions = np.array([c.position for c in self.population])
        colors = [c.color for c in self.population]
        self.creature_plot.set_offsets(positions)
        self.creature_plot.set_color(colors)
        
        avg_sensors = np.mean([len(c.active_sensors) for c in self.population])
        avg_actions = np.mean([len(c.active_actions) for c in self.population])
        in_safe_zone = sum(1 for c in self.population if c.position[0] >= SAFE_ZONE_START)
        
        self.info_text.set_text(
            f"Generation: {self.generation}\n"
            f"Frame: {self.current_frame}/{GENERATION_LENGTH}\n"
            f"Population: {len(self.population)}\n"
            f"In Safe Zone: {in_safe_zone}\n"
            f"Survivors Last Gen: {self.survivors_last_gen}\n"
            f"Avg Sensors: {avg_sensors:.1f}/{len(POSSIBLE_SENSORS)}\n"
            f"Avg Actions: {avg_actions:.1f}/{len(POSSIBLE_ACTIONS)}"
        )
        
        plt.draw()
        plt.pause(0.001)

    def run_frame(self):
        self.current_time += 1
        self.current_frame += 1
        
        self.pheromone_grid *= 0.95
        self._update_grids()
        random.shuffle(self.population)
        
        for creature in self.population:
            creature.update_sensors(
                grid=self.grid,
                current_time=self.current_time,
                pheromone_grid=self.pheromone_grid,
                creature_grid=self.creature_grid
            )
            
            action = creature.decide_action()
            creature.move(self.grid, action, self.pheromone_grid)
        
        self._update_visualization()

    def run_generation(self):
        self.current_frame = 0
        for _ in range(GENERATION_LENGTH):
            self.run_frame()
        
        # Determine survivors based on the current survival zone
        if self.generation < MAX_GENERATIONS // 2:
            # Right part is the safe zone
            survivors = [c for c in self.population if c.position[0] >= SAFE_ZONE_START]
        else:
            # Left part is the safe zone
            survivors = [c for c in self.population if c.position[0] < SAFE_ZONE_START]
        
        self.survivors_last_gen = len(survivors)
        
        # Calculate and store metrics
        survival_rate = (len(survivors) / POPULATION_SIZE) * 100
        genetic_diversity = self._calculate_genetic_diversity()
        
        self.survival_rates.append(survival_rate)
        self.genetic_diversities.append(genetic_diversity)
        self.generation_numbers.append(self.generation)
        
        # Update the safe zone visualization
        self._update_safe_zone_visualization()
        
        if len(survivors) == 0:
            print(f"\nTerminating: No survivors in generation {self.generation}")
            return False
        
        # Create gene pool from all survivors
        gene_pool = []
        for survivor in survivors:
            gene_pool.extend(survivor.dna)
        
        new_population = []
        total_mutations = 0
        
        # Create new population using random gene selection
        for _ in range(POPULATION_CAP):
            # Randomly select genes from the gene pool
            selected_genes = random.choices(gene_pool, k=GENE_LENGTH)
            
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
        
        # Update population statistics
        if survivors:
            avg_mutations = total_mutations / POPULATION_CAP
            print(f"Gen {self.generation}: Survival={survival_rate:.1f}%, Diversity={genetic_diversity}, Mutations={avg_mutations:.3f}/creature")
        
        self.population = new_population
        self.generation += 1
        self._update_grids()
        return True

    def save_metrics_to_csv(self, filename_prefix="simulation_metrics"):
        """Save survival rate and genetic diversity data to a CSV file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
        filename = f"{filename_prefix}_{timestamp}.csv"  # Append timestamp to filename
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Survival Rate (%)", "Genetic Diversity"])
            for gen, survival_rate, diversity in zip(self.generation_numbers, self.survival_rates, self.genetic_diversities):
                writer.writerow([gen, survival_rate, diversity])
        print(f"Metrics saved to {filename}")

    def save_genomes_to_csv(self, filename_prefix="creature_genomes"):
        """Save the genome of all creatures to a CSV file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
        filename = f"{filename_prefix}_{timestamp}.csv"  # Append timestamp to filename
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Creature ID", "Genome"])  # Header row
            for idx, creature in enumerate(self.population):
                # Convert the DNA (list of tuples) to a string for saving
                genome_str = ";".join([f"{gene}" for gene in creature.dna])
                writer.writerow([idx, genome_str])
        print(f"Genomes saved to {filename}")

    def plot_metrics(self):
        plt.ioff()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot survival rate
        ax1.plot(self.generation_numbers, self.survival_rates, 'b-', marker='o')
        ax1.set_title('Survival Rate Over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Survival Rate (%)')
        ax1.grid(True)
        
        # Plot genetic diversity
        ax2.plot(self.generation_numbers, self.genetic_diversities, 'r-', marker='o')
        ax2.set_title('Genetic Diversity Over Generations')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Unique Gene Combinations')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    print(f"Starting evolution simulation with:")
    print(f"- {HIDDEN_NODES} hidden nodes")
    print(f"- Safe zone (right half, x >= {SAFE_ZONE_START})")
    print(f"- Mutation rate: 1 in {int(1/MUTATION_RATE)} base pairs")
    print("Press Ctrl+C to stop")
    
    sim = Simulation(generations_to_track=10)
    try:
        while sim.generation < MAX_GENERATIONS:
            should_continue = sim.run_generation()
            if not should_continue:
                break
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        plt.ioff()
        print(f"\nFinal stats:")
        print(f"- Total generations: {sim.generation}")
        print(f"- Last survivors: {sim.survivors_last_gen}")
        
        # Save metrics and genomes
        sim.save_metrics_to_csv(f"simulation_metrics.csv")
        sim.save_genomes_to_csv(f"creature_genomes.csv")
        sim.plot_metrics()

if __name__ == "__main__":
    main()