import os
import csv
import random
import math
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import original Creature and Villain classes from the main simulation
class Villain:
    """Represents a villain creature that hunts other creatures."""
    
    def __init__(self, width, height):
        self.x = random.randint(50, width - 50)
        self.y = random.randint(50, height - 50)
        self.size = 15
        self.color = "#FF0000"  # Red
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 3.0
        self.hunt_range = 150  # Detection range for hunting
        
    def update(self, creatures, width, height):
        """Update villain position and behavior."""
        if creatures:
            # Find closest creature to hunt
            closest_creature = min(creatures, key=lambda c: 
                               ((c.x - self.x) ** 2 + (c.y - self.y) ** 2) ** 0.5)
            
            distance = ((closest_creature.x - self.x) ** 2 + (closest_creature.y - self.y) ** 2) ** 0.5
            
            # Only hunt if within range
            if distance < self.hunt_range:
                # Calculate direction to creature
                dx = closest_creature.x - self.x
                dy = closest_creature.y - self.y
                
                # Update angle to face creature
                target_angle = math.atan2(dy, dx)
                angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
                
                # Gradually turn toward creature
                self.angle += angle_diff * 0.2
            else:
                # Random movement if no creatures in range
                self.angle += random.uniform(-0.2, 0.2)
        else:
            # Random movement if no creatures
            self.angle += random.uniform(-0.2, 0.2)
        
        # Move in current direction
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # Wrap around edges
        self.x = self.x % width
        self.y = self.y % height
        
    def eat(self, creatures):
        """Check if villain can eat any creatures and return eaten ones."""
        eaten = []
        eat_distance = self.size + 5
        
        for creature in creatures:
            if ((creature.x - self.x) ** 2 + (creature.y - self.y) ** 2) < eat_distance ** 2:
                eaten.append(creature)
                
        return eaten


class Creature:
    """Represents an individual creature in the simulation."""
    
    def __init__(self, genome=None, genome_length=12):  # Extended genome for villain avoidance
        # Create random genome if none provided
        if genome is None:
            self.genome = [random.uniform(-1, 1) for _ in range(genome_length)]
        else:
            self.genome = genome
        
        self.fitness = 0
        self.x = random.randint(50, 750)
        self.y = random.randint(50, 550)
        self.size = 10
        self.color = self._genome_to_color()
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = abs(self.genome[0]) * 5  # First gene controls speed
        self.survival_time = 0  # Time alive contributes to fitness
        
    def _genome_to_color(self):
        """Convert genome to RGB color."""
        r = min(255, max(0, int((self.genome[1] + 1) * 127)))
        g = min(255, max(0, int((self.genome[2] + 1) * 127)))
        b = min(255, max(0, int((self.genome[3] + 1) * 127)))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def update(self, food_items, villain, width, height):
        """Update creature position and behavior based on genome and environment."""
        # Calculate villain influence
        villain_influence = False
        villain_distance = float('inf')
        
        if villain:
            dx_villain = villain.x - self.x
            dy_villain = villain.y - self.y
            villain_distance = max(0.1, (dx_villain ** 2 + dy_villain ** 2) ** 0.5)
            
            # Detect villain within range (using genes 8-11 for villain detection range)
            detection_range = 100 + abs(self.genome[8]) * 50
            if villain_distance < detection_range:
                villain_influence = True
        
        # Handle food seeking (genes 4-7)
        if food_items and not villain_influence:
            # Find closest food
            closest_food = min(food_items, key=lambda food: 
                               ((food[0] - self.x) ** 2 + (food[1] - self.y) ** 2) ** 0.5)
            
            # Calculate direction to food
            dx = closest_food[0] - self.x
            dy = closest_food[1] - self.y
            distance = max(0.1, (dx ** 2 + dy ** 2) ** 0.5)
            
            # Update angle based on genome
            angle_change = (self.genome[4] * math.sin(distance * self.genome[5]) + 
                           self.genome[6] * math.cos(distance * self.genome[7]))
            self.angle += angle_change * 0.2
        elif villain_influence:
            # Flee from villain (genes 8-11)
            # Negative direction from villain (fleeing)
            flee_angle = math.atan2(-dy_villain, -dx_villain)
            
            # How much to adjust course based on villain proximity
            adjustment = self.genome[9] * (1 - min(1, villain_distance / 200))
            
            # Add some randomness to escape pattern based on genes
            flee_randomness = self.genome[10] * math.sin(villain_distance * self.genome[11])
            
            # Adjust angle toward flee direction plus randomness
            angle_diff = (flee_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += angle_diff * adjustment + flee_randomness * 0.3
            
            # Boost speed when fleeing
            speed_boost = 1.0 + abs(self.genome[8]) * (1 - min(1, villain_distance / 150))
        else:
            # Random movement if no food or villain
            self.angle += random.uniform(-0.5, 0.5)
        
        # Move in current direction
        effective_speed = self.speed * (1.5 if villain_influence else 1.0)
        self.x += math.cos(self.angle) * effective_speed
        self.y += math.sin(self.angle) * effective_speed
        
        # Wrap around edges
        self.x = self.x % width
        self.y = self.y % height
        
        # Increment survival time
        self.survival_time += 1
        
    def eat(self, food_items):
        """Check if creature can eat any food and return eaten items."""
        eaten = []
        eat_distance = self.size + 5
        
        for food in food_items:
            if ((food[0] - self.x) ** 2 + (food[1] - self.y) ** 2) < eat_distance ** 2:
                eaten.append(food)
                self.fitness += 1
                
        return eaten
    
    def mutate(self, mutation_rate=0.1, mutation_amount=0.2):
        """Create a mutated copy of this creature."""
        new_genome = self.genome.copy()
        
        for i in range(len(new_genome)):
            if random.random() < mutation_rate:
                new_genome[i] += random.uniform(-mutation_amount, mutation_amount)
                new_genome[i] = max(-1, min(1, new_genome[i]))  # Keep within bounds
                
        return Creature(genome=new_genome)
    
    def crossover(self, other):
        """Create offspring by crossing over genes with another creature."""
        crossover_point = random.randint(1, len(self.genome) - 1)
        new_genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        return Creature(genome=new_genome)


class SimulationData:
    """Stores and manages data from simulation runs."""
    
    def __init__(self):
        self.current_run = 0
        self.max_runs = 30
        self.current_generation = 0
        self.max_generations = 50
        
        # Data for current run
        self.generation_data = []  # List of dicts for each generation
        
        # Summary data across all runs
        self.run_summaries = []  # List of summary dicts for each run
        
        # Output directory
        self.output_dir = "simulation_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def record_generation(self, generation, best_fitness, avg_fitness, eaten, duration):
        """Record metrics for the current generation."""
        self.generation_data.append({
            'Generation': generation,
            'BestFitness': best_fitness,
            'AvgFitness': avg_fitness,
            'Eaten': eaten,
            'Duration': duration
        })
        self.current_generation = generation
    
    def finish_run(self, plateau_generation=None):
        """Finalize the current run and save data."""
        if not self.generation_data:
            return
        
        # Calculate summary metrics
        max_best_fitness = max(gen['BestFitness'] for gen in self.generation_data)
        total_generations = len(self.generation_data)
        
        # If plateau wasn't detected during simulation, try to detect it now
        if plateau_generation is None:
            plateau_generation = self._detect_plateau()
        
        # Create run summary
        run_summary = {
            'Run': self.current_run,
            'MaxBestFitness': max_best_fitness,
            'PlateauGeneration': plateau_generation if plateau_generation else total_generations,
            'TotalGenerations': total_generations
        }
        self.run_summaries.append(run_summary)
        
        # Save this run's data to CSV
        self._save_run_to_csv()
        
        # Clear generation data for next run
        self.generation_data = []
        self.current_generation = 0
        self.current_run += 1
    
    def _detect_plateau(self, window_size=5, threshold=0.05):
        """Detect when fitness has plateaued."""
        if len(self.generation_data) < window_size * 2:
            return None
        
        best_fitness_values = [gen['BestFitness'] for gen in self.generation_data]
        
        for i in range(window_size, len(best_fitness_values) - window_size + 1):
            window1 = best_fitness_values[i-window_size:i]
            window2 = best_fitness_values[i:i+window_size]
            
            avg1 = sum(window1) / len(window1)
            avg2 = sum(window2) / len(window2)
            
            # If improvement is less than threshold percentage
            if avg1 > 0 and abs((avg2 - avg1) / avg1) < threshold:
                return i
        
        return None
    
    def _save_run_to_csv(self):
        """Save the current run's generation data to a CSV file."""
        filename = os.path.join(self.output_dir, f"run_{self.current_run}_data.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Generation', 'BestFitness', 'AvgFitness', 'Eaten', 'Duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for gen_data in self.generation_data:
                writer.writerow(gen_data)
    
    def save_summary_to_csv(self):
        """Save summary data for all runs to a CSV file."""
        if not self.run_summaries:
            return
            
        filename = os.path.join(self.output_dir, "simulation_summary.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Run', 'MaxBestFitness', 'PlateauGeneration', 'TotalGenerations']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for summary in self.run_summaries:
                writer.writerow(summary)
    
    def plot_fitness_comparison(self):
        """Create and save a plot comparing fitness across runs."""
        if not self.run_summaries:
            return
            
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        runs = [summary['Run'] for summary in self.run_summaries]
        max_fitness = [summary['MaxBestFitness'] for summary in self.run_summaries]
        plateau_gens = [summary['PlateauGeneration'] for summary in self.run_summaries]
        
        plt.subplot(1, 2, 1)
        plt.bar(runs, max_fitness)
        plt.xlabel('Run')
        plt.ylabel('Max Best Fitness')
        plt.title('Maximum Fitness per Run')
        
        plt.subplot(1, 2, 2)
        plt.bar(runs, plateau_gens)
        plt.xlabel('Run')
        plt.ylabel('Generation')
        plt.title('Plateau Generation per Run')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fitness_comparison.png"))
        plt.close()


# Modifications to GeneticSimulator class
class GeneticSimulator:
    """Manages the genetic algorithm and simulation."""
    
    def __init__(self, canvas_width, canvas_height, population_size=20, 
                data_collector=None, max_generations=50, epsilon=0.05):
        self.width = canvas_width
        self.height = canvas_height
        self.population_size = population_size
        self.creatures = [Creature() for _ in range(population_size)]
        self.food_items = []
        self.generation = 1
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.eaten_history = []  # Track creatures eaten by villain
        self.length_history = []  # Track duration of each generation
        self.time_step = 0
        self.add_food(20)
        self.villain = Villain(canvas_width, canvas_height)
        self.creatures_eaten = 0
        
        # Data collection
        self.data_collector = data_collector
        self.max_generations = max_generations
        self.epsilon = epsilon  # Threshold for plateau detection
        self.plateau_detected = False
    
    def add_food(self, count):
        """Add food items to the simulation."""
        for _ in range(count):
            x = random.randint(10, self.width - 10)
            y = random.randint(10, self.height - 10)
            self.food_items.append((x, y))
    
    def update(self):
        """Update simulation by one time step."""
        self.time_step += 1
        
        # Update villain
        self.villain.update(self.creatures, self.width, self.height)
        
        # Check if villain eats any creatures
        eaten_creatures = self.villain.eat(self.creatures)
        for creature in eaten_creatures:
            if creature in self.creatures:  # Safety check
                self.creatures.remove(creature)
                self.creatures_eaten += 1
        
        # Update all remaining creatures
        for creature in self.creatures:
            creature.update(self.food_items, self.villain, self.width, self.height)
            
            # Handle eating
            eaten = creature.eat(self.food_items)
            for food in eaten:
                self.food_items.remove(food)
        
        # Award survival fitness
        for creature in self.creatures:
            # Small fitness boost for surviving (less than eating food)
            creature.fitness += 0.01
        
        # Add more food periodically
        if self.time_step % 50 == 0:
            self.add_food(5)
            
        # Check if generation should end
        if self.time_step >= 500 or not self.food_items or not self.creatures:
            self.evolve()
            return True
        
        return False
    
    def evolve(self):
        """Move to next generation using genetic algorithm."""
        # Record generation data
        best_fitness = 0
        avg_fitness = 0
        
        # Calculate fitness
        if self.creatures:
            total_fitness = sum(creature.fitness for creature in self.creatures)
            
            if total_fitness > 0:
                # Record statistics
                best_fitness = max(creature.fitness for creature in self.creatures)
                avg_fitness = total_fitness / len(self.creatures)
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                self.eaten_history.append(self.creatures_eaten)
                self.length_history.append(self.time_step)
                
                # Check for plateau
                if len(self.best_fitness_history) >= 5:  # Need at least 5 generations
                    self._check_plateau()
                
                # Save generation data if collector is available
                if self.data_collector:
                    self.data_collector.record_generation(
                        self.generation, best_fitness, avg_fitness, 
                        self.creatures_eaten, self.time_step
                    )
                
                # Select parents based on fitness (roulette wheel selection)
                parents = []
                for _ in range(self.population_size):
                    pick = random.uniform(0, total_fitness)
                    current = 0
                    for creature in self.creatures:
                        current += creature.fitness
                        if current >= pick:
                            parents.append(creature)
                            break
                    else:
                        # Fallback if rounding errors occur
                        if self.creatures:
                            parents.append(random.choice(self.creatures))
                
                # Create new generation with crossover and mutation
                new_generation = []
                
                # Keep the best creature if available (elitism)
                if self.creatures:
                    best_creature = max(self.creatures, key=lambda c: c.fitness)
                    new_generation.append(Creature(genome=best_creature.genome))
                
                # Fill rest with offspring
                while len(new_generation) < self.population_size:
                    if len(parents) >= 2:
                        parent1 = random.choice(parents)
                        parent2 = random.choice(parents)
                        
                        if random.random() < 0.7:  # 70% chance of crossover
                            offspring = parent1.crossover(parent2)
                        else:
                            # Copy from the better parent
                            better_parent = parent1 if parent1.fitness > parent2.fitness else parent2
                            offspring = Creature(genome=better_parent.genome)
                        
                        # Mutate
                        offspring = offspring.mutate()
                        new_generation.append(offspring)
                    else:
                        # Not enough parents, add random creature
                        new_generation.append(Creature())
            else:
                # If all fitness is zero, create fresh generation
                self.best_fitness_history.append(0)
                self.avg_fitness_history.append(0)
                self.eaten_history.append(self.creatures_eaten)
                self.length_history.append(self.time_step)
                
                if self.data_collector:
                    self.data_collector.record_generation(
                        self.generation, 0, 0, self.creatures_eaten, self.time_step
                    )
                
                new_generation = [Creature() for _ in range(self.population_size)]
        else:
            # All creatures eaten, create fresh generation
            self.best_fitness_history.append(0)
            self.avg_fitness_history.append(0)
            self.eaten_history.append(self.creatures_eaten)
            self.length_history.append(self.time_step)
            
            if self.data_collector:
                self.data_collector.record_generation(
                    self.generation, 0, 0, self.creatures_eaten, self.time_step
                )
            
            new_generation = [Creature() for _ in range(self.population_size)]
        
        # Replace old generation
        self.creatures = new_generation
        
        # Reset for next generation
        self.generation += 1
        self.time_step = 0
        self.food_items = []
        self.add_food(20)
        
        # Reset villain position
        self.villain = Villain(self.width, self.height)
        self.creatures_eaten = 0
    
    def _check_plateau(self, window=5):
        """Check if fitness has plateaued."""
        if self.plateau_detected or len(self.best_fitness_history) < window * 2:
            return False
            
        # Get recent fitness values
        recent = self.best_fitness_history[-window:]
        previous = self.best_fitness_history[-(window*2):-window]
        
        # Calculate average improvement
        avg_recent = sum(recent) / window
        avg_previous = sum(previous) / window
        
        # Check if improvement is below threshold
        if avg_previous > 0 and abs((avg_recent - avg_previous) / avg_previous) < self.epsilon:
            self.plateau_detected = True
            return True
            
        return False
    
    def should_terminate(self):
        """Check if simulation should terminate based on generations or plateau."""
        if self.generation > self.max_generations:
            return True
            
        if self.plateau_detected:
            return True
            
        return False
    
    def get_plateau_generation(self):
        """Return the generation where plateau was detected, or None."""
        if self.plateau_detected:
            # Find the generation where plateau was first detected
            # This is approximate based on when _check_plateau returned True
            return max(1, self.generation - 5)
        return None


# BatchSimulationRunner to handle multiple simulation runs
class BatchSimulationRunner:
    """Manages batch execution of multiple simulation runs."""
    
    def __init__(self, canvas_width, canvas_height, num_runs=30, 
                population_size=20, max_generations=50):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.num_runs = num_runs
        self.population_size = population_size
        self.max_generations = max_generations
        
        self.data_collector = SimulationData()
        self.data_collector.max_runs = num_runs
        self.data_collector.max_generations = max_generations
        
        self.current_run = 0
        self.running = False
        self.paused = False
        
        # Current simulator instance
        self.simulator = None
        
        # GUI callback functions
        self.on_progress_update = None
        self.on_run_complete = None
        self.on_batch_complete = None
    
    def start(self):
        """Start the batch simulation."""
        if self.running:
            return
            
        self.running = True
        self.current_run = 0
        
        # Start simulation thread
        thread = threading.Thread(target=self._run_batch)
        thread.daemon = True
        thread.start()
    
    def pause(self):
        """Pause the batch simulation."""
        self.paused = not self.paused
        return self.paused
    
    def stop(self):
        """Stop the batch simulation."""
        self.running = False
    
    def _run_batch(self):
        """Run all simulations in the batch."""
        self.data_collector.current_run = 0
        
        for run in range(self.num_runs):
            if not self.running:
                break
                
            self.current_run = run
            self.data_collector.current_run = run
            
            # Initialize new simulator
            self.simulator = GeneticSimulator(
                self.canvas_width, self.canvas_height,
                population_size=self.population_size,
                data_collector=self.data_collector,
                max_generations=self.max_generations
            )
            
            # Run this simulation
            self._run_single_simulation()
            
            # Finalize data for this run
            self.data_collector.finish_run(self.simulator.get_plateau_generation())
            
            # Notify of run completion
            if self.on_run_complete:
                self.on_run_complete(run)
        
        # Save summary data when all runs complete
        self.data_collector.save_summary_to_csv()
        self.data_collector.plot_fitness_comparison()
        
        # Notify of batch completion
        if self.on_batch_complete:
            self.on_batch_complete()
        
        self.running = False
    
    def _run_single_simulation(self):
        """Run a single simulation until termination."""
        generation_count = 0
        
        while self.running and not self.simulator.should_terminate():
            # Handle pause
            while self.paused and self.running:
                time.sleep(0.1)
                
            if not self.running:
                break
                
            # Run a full generation
            while not self.simulator.update():
                # Update progress more frequently for responsive UI
                if self.simulator.time_step % 50 == 0 and self.on_progress_update:
                    self.on_progress_update(
                        self.current_run, self.simulator.generation, 
                        self.simulator.time_step, False
                    )
                    
                # Small sleep to prevent CPU hogging
                time.sleep(0.001)
            
            # Generation complete
            generation_count += 1
            
            # Update progress after each generation
            if self.on_progress_update:
                self.on_progress_update(
                    self.current_run, self.simulator.generation, 
                    self.simulator.time_step, True
                )


# GUI extension for batch simulation
class BatchSimulationApp:
    """GUI application for running batch simulations."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Creature Evolution Batch Simulator")
        
        # Set up batch runner
        self.canvas_width = 800
        self.canvas_height = 600
        self.batch_runner = BatchSimulationRunner(
            self.canvas_width, self.canvas_height
        )
        
        # Connect callbacks
        self.batch_runner.on_progress_update = self.update_progress
        self.batch_runner.on_run_complete = self.on_run_complete
        self.batch_runner.on_batch_complete = self.on_batch_complete
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Create the UI for batch simulation control."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Simulation Parameters", padding="10")
        params_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Number of runs
        ttk.Label(params_frame, text="Number of Runs:").grid(row=0, column=0, sticky="w")
        self.runs_var = tk.StringVar(value="30")
        runs_entry = ttk.Entry(params_frame, textvariable=self.runs_var, width=5)
        runs_entry.grid(row=0, column=1, sticky="w", padx=5)
        
        # Population size
        ttk.Label(params_frame, text="Population Size:").grid(row=1, column=0, sticky="w")
        self.pop_var = tk.StringVar(value="20")
        pop_entry = ttk.Entry(params_frame, textvariable=self.pop_var, width=5)
        pop_entry.grid(row=1, column=1, sticky="w", padx=5)
        
        # Max generations
        ttk.Label(params_frame, text="Max Generations:").grid(row=2, column=0, sticky="w")
        self.gen_var = tk.StringVar(value="50")
        gen_entry = ttk.Entry(params_frame, textvariable=self.gen_var, width=5)
        gen_entry.grid(row=2, column=1, sticky="w", padx=5)
        
        # Output directory
        ttk.Label(params_frame, text="Output Directory:").grid(row=3, column=0, sticky="w")
        self.dir_var = tk.StringVar(value="simulation_results")
        dir_entry = ttk.Entry(params_frame, textvariable=self.dir_var, width=30)
        dir_entry.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        dir_button = ttk.Button(params_frame, text="Browse...", command=self.browse_directory)
        dir_button.grid(row=3, column=3, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Batch", command=self.start_batch)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_batch, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_batch, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:").grid(row=0, column=0, sticky="w")
        self.overall_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.overall_progress.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Current run progress
        ttk.Label(progress_frame, text="Current Run:").grid(row=1, column=0, sticky="w")
        self.run_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.run_progress.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Current generation progress
        ttk.Label(progress_frame, text="Current Generation:").grid(row=2, column=0, sticky="w")
        self.gen_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.gen_progress.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to start")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=1)
    
    def browse_directory(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_var.set(directory)
            self.batch_runner.data_collector.output_dir = directory
    
    def start_batch(self):
        """Start the batch simulation."""
        try:
            num_runs = int(self.runs_var.get())
            pop_size = int(self.pop_var.get())
            max_gens = int(self.gen_var.get())
            
            if num_runs < 1 or pop_size < 2 or max_gens < 1:
                messagebox.showerror("Invalid Parameters", 
                                     "All parameters must be positive numbers.")
                return
                
            # Update batch runner parameters
            self.batch_runner.num_runs = num_runs
            self.batch_runner.population_size = pop_size
            self.batch_runner.max_generations = max_gens
            self.batch_runner.data_collector.max_runs = num_runs
            self.batch_runner.data_collector.max_generations = max_gens
            self.batch_runner.data_collector.output_dir = self.dir_var.get()
            
            # Create output directory if it doesn't exist
            os.makedirs(self.dir_var.get(), exist_ok=True)
            
            # Start simulation
            self.batch_runner.start()
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            
            # Reset progress bars
            self.overall_progress['maximum'] = num_runs
            self.overall_progress['value'] = 0
            self.run_progress['maximum'] = max_gens
            self.run_progress['value'] = 0
            self.gen_progress['maximum'] = 500  # Default generation length
            self.gen_progress['value'] = 0
            
            self.status_var.set(f"Starting batch simulation: {num_runs} runs")
            
        except ValueError:
            messagebox.showerror("Invalid Parameters", 
                                "All parameters must be valid numbers.")
    
    def pause_batch(self):
        """Pause or resume the batch simulation."""
        is_paused = self.batch_runner.pause()
        if is_paused:
            self.pause_button.config(text="Resume")
            self.status_var.set("Simulation paused")
        else:
            self.pause_button.config(text="Pause")
            self.status_var.set("Simulation running")
    
    def stop_batch(self):
        """Stop the batch simulation."""
        self.batch_runner.stop()
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Simulation stopped")
    
    def update_progress(self, run, generation, time_step, gen_complete):
        """Update progress bars based on simulation progress."""
        # Update overall progress
        self.overall_progress['value'] = run
        
        # Update run progress (generations)
        self.run_progress['value'] = generation
        
        # Update generation progress
        if gen_complete:
            self.gen_progress['value'] = 500  # Complete
        else:
            self.gen_progress['value'] = time_step
        
        self.status_var.set(f"Run {run+1}/{self.batch_runner.num_runs}, "
                           f"Generation {generation}/{self.batch_runner.max_generations}, "
                           f"Time step: {time_step}")
        
        # Force UI update
        self.root.update_idletasks()
    
    def on_run_complete(self, run):
        """Handler for completion of a single run."""
        self.overall_progress['value'] = run + 1
        self.run_progress['value'] = 0
        self.gen_progress['value'] = 0
        self.status_var.set(f"Completed run {run+1}/{self.batch_runner.num_runs}")
        self.root.update_idletasks()
    
    def on_batch_complete(self):
        """Handler for completion of the entire batch."""
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        
        self.status_var.set(f"Batch complete! Results saved to {self.dir_var.get()}")
        
        messagebox.showinfo("Batch Complete", 
                           f"All {self.batch_runner.num_runs} simulation runs completed.\n"
                           f"Results saved to {self.dir_var.get()}")


# Add original visualization app
class EvolutionApp:
    """Main application for the evolution simulator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Creature Evolution Simulator with Villain")
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.setup_controls()
        
        # Create the canvas for visualization
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_width, 
                               height=self.canvas_height, bg="black")
        self.canvas.grid(row=0, column=1, rowspan=10, padx=10, pady=10, sticky="nsew")
        
        # Set up plotting area for fitness history
        self.setup_plots()
        
        # Initialize simulation
        self.running = False
        self.simulator = GeneticSimulator(self.canvas_width, self.canvas_height,
                                         population_size=int(self.population_var.get()))
        self.speed = 20  # ms between updates
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Start the simulation
        self.update_display()
    
    def setup_controls(self):
        """Create control panel with sliders and buttons."""
        control_frame = ttk.LabelFrame(self.main_frame, text="Simulation Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)
        
        # Population size
        ttk.Label(control_frame, text="Population Size:").grid(row=0, column=0, sticky="w")
        self.population_var = tk.StringVar(value="20")
        population_entry = ttk.Entry(control_frame, textvariable=self.population_var, width=5)
        population_entry.grid(row=0, column=1, sticky="w")
        
        # Mutation rate
        ttk.Label(control_frame, text="Mutation Rate:").grid(row=1, column=0, sticky="w")
        self.mutation_rate = tk.DoubleVar(value=0.1)
        mutation_slider = ttk.Scale(control_frame, from_=0.01, to=0.5, 
                                   variable=self.mutation_rate, orient=tk.HORIZONTAL)
        mutation_slider.grid(row=1, column=1, sticky="we")
        
        # Speed control
        ttk.Label(control_frame, text="Simulation Speed:").grid(row=2, column=0, sticky="w")
        self.sim_speed = tk.IntVar(value=20)
        speed_slider = ttk.Scale(control_frame, from_=1, to=50, 
                                variable=self.sim_speed, orient=tk.HORIZONTAL,
                                command=self.update_speed)
        speed_slider.grid(row=2, column=1, sticky="we")
        
        # Add food button
        ttk.Button(control_frame, text="Add Food", 
                  command=lambda: self.simulator.add_food(10)).grid(row=3, column=0, pady=5)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_simulation)
        self.start_button.grid(row=3, column=1, pady=5)
        
        # Reset button
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).grid(
            row=4, column=0, columnspan=2, pady=5)
        
        # Generation and fitness info
        info_frame = ttk.LabelFrame(self.main_frame, text="Simulation Info", padding="10")
        info_frame.grid(row=1, column=0, sticky="nw", padx=10, pady=10)
        
        ttk.Label(info_frame, text="Generation:").grid(row=0, column=0, sticky="w")
        self.generation_label = ttk.Label(info_frame, text="1")
        self.generation_label.grid(row=0, column=1, sticky="w")
        
        ttk.Label(info_frame, text="Best Fitness:").grid(row=1, column=0, sticky="w")
        self.best_fitness_label = ttk.Label(info_frame, text="0")
        self.best_fitness_label.grid(row=1, column=1, sticky="w")
        
        ttk.Label(info_frame, text="Avg Fitness:").grid(row=2, column=0, sticky="w")
        self.avg_fitness_label = ttk.Label(info_frame, text="0")
        self.avg_fitness_label.grid(row=2, column=1, sticky="w")
        
        ttk.Label(info_frame, text="Food Left:").grid(row=3, column=0, sticky="w")
        self.food_label = ttk.Label(info_frame, text="20")
        self.food_label.grid(row=3, column=1, sticky="w")
        
        ttk.Label(info_frame, text="Creatures Eaten:").grid(row=4, column=0, sticky="w")
        self.eaten_label = ttk.Label(info_frame, text="0")
        self.eaten_label.grid(row=4, column=1, sticky="w")
    
    def setup_plots(self):
        """Create plot area for fitness history."""
        plot_frame = ttk.LabelFrame(self.main_frame, text="Fitness History", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nw", padx=10, pady=10)
        
        # Create figure and canvas for plotting
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot.set_xlabel("Generation")
        self.plot.set_ylabel("Fitness")
        self.plot.grid(True)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack()
    
    def update_display(self):
        """Update the canvas and UI elements."""
        if self.running:
            # Run simulation step
            generation_ended = self.simulator.update()
            
            if generation_ended:
                self.update_plots()
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Draw food
            for food in self.simulator.food_items:
                self.canvas.create_oval(food[0]-3, food[1]-3, food[0]+3, food[1]+3, 
                                       fill="green", outline="")
            
            # Draw creatures
            for creature in self.simulator.creatures:
                self.canvas.create_oval(creature.x - creature.size, creature.y - creature.size,
                                       creature.x + creature.size, creature.y + creature.size,
                                       fill=creature.color, outline="white")
                
                # Draw direction indicator
                direction_x = creature.x + math.cos(creature.angle) * creature.size * 1.5
                direction_y = creature.y + math.sin(creature.angle) * creature.size * 1.5
                self.canvas.create_line(creature.x, creature.y, direction_x, direction_y, 
                                       fill="white", width=2)
            
            # Draw villain
            villain = self.simulator.villain
            self.canvas.create_oval(villain.x - villain.size, villain.y - villain.size,
                                   villain.x + villain.size, villain.y + villain.size,
                                   fill=villain.color, outline="black", width=2)
            
            # Draw villain direction
            v_direction_x = villain.x + math.cos(villain.angle) * villain.size * 1.5
            v_direction_y = villain.y + math.sin(villain.angle) * villain.size * 1.5
            self.canvas.create_line(villain.x, villain.y, v_direction_x, v_direction_y, 
                                   fill="black", width=2)
            
            # Draw villain hunt range (optional visualization)
            self.canvas.create_oval(villain.x - villain.hunt_range, villain.y - villain.hunt_range,
                                   villain.x + villain.hunt_range, villain.y + villain.hunt_range,
                                   outline="red", dash=(2, 4))
            
            # Update info labels
            self.generation_label.config(text=str(self.simulator.generation))
            
            if self.simulator.creatures:
                best_fitness = max(creature.fitness for creature in self.simulator.creatures)
                avg_fitness = sum(creature.fitness for creature in self.simulator.creatures) / len(self.simulator.creatures)
                self.best_fitness_label.config(text=f"{best_fitness:.2f}")
                self.avg_fitness_label.config(text=f"{avg_fitness:.2f}")
            
            self.food_label.config(text=str(len(self.simulator.food_items)))
            self.eaten_label.config(text=str(self.simulator.creatures_eaten))
            
        # Schedule next update
        self.root.after(self.speed, self.update_display)
    
    def update_plots(self):
        """Update the fitness history plots."""
        self.plot.clear()
        
        generations = list(range(1, len(self.simulator.best_fitness_history) + 1))
        
        if generations:
            self.plot.plot(generations, self.simulator.best_fitness_history, 
                          label="Best Fitness", color="red", marker="o")
            self.plot.plot(generations, self.simulator.avg_fitness_history, 
                          label="Avg Fitness", color="blue", marker=".")
            
            self.plot.set_xlabel("Generation")
            self.plot.set_ylabel("Fitness")
            self.plot.legend()
            self.plot.grid(True)
            
            self.canvas_plot.draw()
    
    def toggle_simulation(self):
        """Start or stop the simulation."""
        self.running = not self.running
        
        if self.running:
            self.start_button.config(text="Pause")
        else:
            self.start_button.config(text="Resume")
    
    def update_speed(self, *args):
        """Update simulation speed based on slider."""
        self.speed = 51 - self.sim_speed.get()  # Invert so higher = faster
    
    def reset_simulation(self):
        """Reset the entire simulation."""
        try:
            pop_size = int(self.population_var.get())
            if pop_size < 2:
                pop_size = 2
                self.population_var.set("2")
            elif pop_size > 100:
                pop_size = 100
                self.population_var.set("100")
        except ValueError:
            pop_size = 20
            self.population_var.set("20")
            
        self.simulator = GeneticSimulator(self.canvas_width, self.canvas_height, 
                                         population_size=pop_size)
        
        # Reset UI elements
        self.generation_label.config(text="1")
        self.best_fitness_label.config(text="0")
        self.avg_fitness_label.config(text="0")
        self.food_label.config(text="20")
        self.eaten_label.config(text="0")
        
        # Clear plot
        self.plot.clear()
        self.plot.set_xlabel("Generation")
        self.plot.set_ylabel("Fitness")
        self.plot.grid(True)
        self.canvas_plot.draw()


# Modify main function to offer both visualization and batch modes
def main():
    """Run the application."""
    root = tk.Tk()
    
    # Ask user which mode to run
    mode = messagebox.askquestion("Start Mode", 
                                 "Do you want to run in batch data collection mode?\n\n"
                                 "Yes: Run multiple simulations and collect data\n"
                                 "No: Run interactive visualization")
    
    if mode == 'yes':
        # Batch data collection mode
        app = BatchSimulationApp(root)
    else:
        # Interactive visualization mode
        app = EvolutionApp(root)
    
    root.geometry("1200x700")
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
