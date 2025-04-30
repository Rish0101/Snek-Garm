import tkinter as tk
import random
import math
import time
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


class GeneticSimulator:
    """Manages the genetic algorithm and simulation."""
    
    def __init__(self, canvas_width, canvas_height, population_size=20):
        self.width = canvas_width
        self.height = canvas_height
        self.population_size = population_size
        self.creatures = [Creature() for _ in range(population_size)]
        self.food_items = []
        self.generation = 1
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.time_step = 0
        self.add_food(20)
        self.villain = Villain(canvas_width, canvas_height)
        self.creatures_eaten = 0
        
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
        # Calculate fitness
        if self.creatures:
            total_fitness = sum(creature.fitness for creature in self.creatures)
            
            if total_fitness > 0:
                # Record statistics
                best_fitness = max(creature.fitness for creature in self.creatures)
                avg_fitness = total_fitness / len(self.creatures)
                self.best_fitness_history.append(best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                
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
                new_generation = [Creature() for _ in range(self.population_size)]
        else:
            # All creatures eaten, create fresh generation
            self.best_fitness_history.append(0)
            self.avg_fitness_history.append(0)
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


def main():
    """Run the application."""
    root = tk.Tk()
    app = EvolutionApp(root)
    root.geometry("1200x700")
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
