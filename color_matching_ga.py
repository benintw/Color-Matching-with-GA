"""
A genetic algorithm implementation for evolving image patterns to match target colors.

This module implements a genetic algorithm that evolves a grid of RGB color values
to match a target color. Each individual in the population represents a grid of
RGB values (e.g., 4x4 grid) where the average color aims to match the target.

The algorithm uses:
- Roulette wheel selection based on fitness
- Single-point crossover for reproduction
- Gaussian mutation on individual RGB values
- Elitism to preserve the best solution

Classes:
    GAConfig: Configuration parameters for the genetic algorithm
    ColorMatchingGA: Main genetic algorithm implementation
    Visualizer: Visualization tools for tracking evolution progress

Example:
    config = GAConfig(
        popsize=1000,
        target_value=np.array([222, 165, 33])  # Golden color
    )
    ga = ColorMatchingGA(config)
    best_chromosomes, best_fitness = ga.evolve()
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from datetime import datetime


@dataclass
class GAConfig:
    """Configuration parameters for the genetic algorithm.

    Attributes:
        target_value (np.ndarray): Target RGB color to match as [R,G,B] array
        popsize (int): Number of individuals in each generation
        chromosome_dim (tuple[int, int]): Dimensions of the color grid (height, width)
        max_gen (int): Maximum number of generations to evolve
        p_crossover (float): Probability of performing crossover between parents [0,1]
        p_mutation (float): Probability of mutation for each offspring [0,1]
        early_stopping_patience (int): Number of generations without improvement before stopping
        mutation_std (float): Standard deviation for Gaussian mutation on RGB values
    """

    target_value: np.ndarray
    popsize: int = 100
    chromosome_dim: tuple[int, int] = (4, 4)
    max_gen: int = 100
    p_crossover: float = 0.5
    p_mutation: float = 0.1
    early_stopping_patience: int = 50
    mutation_std: float = 25


class ColorMatchingGA:
    """Genetic Algorithm implementation for color matching.

    This class implements a genetic algorithm to evolve RGB colors
    to match a target color using selection, crossover, and mutation
    operators.

    Attributes:
        config: GAConfig object containing algorithm parameters
        population: Current generation of chromosomes
        best_chromosomes: History of best chromosomes per generation
        best_fitness_values: History of best fitness values per generation
    """

    def __init__(self, config: GAConfig):
        """Initialize the genetic algorithm.

        Args:
            config: GAConfig object containing algorithm parameters
        """
        self.config = config
        self.population: np.ndarray = self._init_population()
        self.best_chromosomes: list[np.ndarray] = []
        self.best_fitness_values: list[float] = []
        self.current_generation: int = 0
        self.best_fitness: float = float("inf")
        self.generation_without_improvement: int = 0

    def _init_population(self) -> np.ndarray:
        """Initialize population with random RGB values.

        Creates a population of random chromosomes where each chromosome is a grid of RGB values.
        Each RGB value is randomly initialized between 0 and 255.

        Returns:
            np.ndarray: Initial population array of shape (popsize, height, width, 3) where:
                - popsize: number of individuals in the population
                - height, width: dimensions of the color grid
                - 3: represents RGB channels
        """
        return np.random.randint(
            0, 256, size=(self.config.popsize, *self.config.chromosome_dim, 3)
        )

    def _calc_fitness(self, chromosomes: np.ndarray) -> np.ndarray:
        """Calculate fitness using squared difference from target.

        Computes the mean squared error between each chromosome's RGB values and the target color.
        Lower fitness values indicate better solutions (closer to target color).

        Args:
            chromosomes: Array of shape (popsize, height, width, 3) containing RGB values
                for each individual in the population

        Returns:
            np.ndarray: Array of shape (popsize,) containing fitness values for each chromosome.
                Each value represents the mean squared error from the target color.
        """

        return np.mean(
            np.square(chromosomes - self.config.target_value), axis=(1, 2, 3)
        )

    def _generate_mating_pool(self, fitness: np.ndarray) -> np.ndarray:
        """Select individuals using Roulette Wheel Selection.

        Implements fitness proportionate selection where individuals with better fitness
        (lower values) have a higher probability of being selected for reproduction.

        Args:
            fitness: Array of shape (popsize,) containing fitness values for current population

        Returns:
            np.ndarray: Selected parents array of shape (popsize, height, width, 3).
                These parents will be used to create the next generation.
        """

        # convert fitness to selection probabilities (lower fitness -> higher prob)
        selection_probs: np.ndarray = 1 / (fitness + 1e-10)
        selection_probs /= selection_probs.sum()

        # select indices based on probabilities
        selected_indices: np.ndarray = np.random.choice(
            a=len(self.population),
            size=len(self.population),
            p=selection_probs,
            replace=True,
        )

        return self.population[selected_indices]

    def _create_offspring(self, mating_pool: np.ndarray) -> np.ndarray:
        """Create new population through crossover and mutation.

        Repeatedly selects pairs of parents from the mating pool to create offspring
        through crossover and mutation until a new population is formed.
        Excludes space for one elite individual.

        Args:
            mating_pool: Array of shape (popsize, height, width, 3) containing
                selected parent chromosomes

        Returns:
            np.ndarray: New population array of shape (popsize-1, height, width, 3).
                Space for one elite individual is left out.
        """
        new_population = np.empty((0, *self.config.chromosome_dim, 3))

        while len(new_population) < self.config.popsize - 1:
            # Select parent pairs
            idx1, idx2 = np.random.choice(len(mating_pool), size=2, replace=False)
            parent1, parent2 = mating_pool[idx1], mating_pool[idx2]

            # Create and mutate offspring
            offspring1, offspring2 = self._crossover(parent1, parent2)
            offspring1, offspring2 = self._mutate(offspring1), self._mutate(offspring2)

            # Add to new population
            new_population = np.vstack(
                (
                    new_population,
                    np.expand_dims(offspring1, 0),
                    np.expand_dims(offspring2, 0),
                )
            )

        return new_population[: self.config.popsize - 1]

    def _update_statistics(self, fitness: np.ndarray) -> tuple[np.ndarray, float]:
        """Update best chromosome and fitness statistics.

        Identifies the elite (best) individual in the current generation and
        updates the historical tracking of best chromosomes and fitness values.

        Args:
            fitness: Array of shape (popsize,) containing fitness values for current population

        Returns:
            tuple containing:
                - np.ndarray: Elite chromosome of shape (height, width, 3)
                - float: Fitness value of the elite chromosome
        """
        elite_idx = np.argmin(fitness)
        elite = self.population[elite_idx]
        elite_fitness = fitness[elite_idx]

        self.best_chromosomes.append(elite)
        self.best_fitness_values.append(elite_fitness)

        return elite, elite_fitness

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform single-point crossover between two parents.

        With probability p_crossover, performs crossover by:
        1. Flattening both parents into 1D arrays
        2. Randomly selecting a crossover point
        3. Swapping genetic material after the crossover point
        If no crossover occurs, returns copies of the parents.

        Args:
            parent1: First parent's chromosome of shape (height, width, 3)
            parent2: Second parent's chromosome of shape (height, width, 3)

        Returns:
            tuple containing two offspring chromosomes, each of shape (height, width, 3)
        """
        if np.random.rand() < self.config.p_crossover:
            crossover_point = parent1.reshape(-1, 3).shape[0] // 2
            parent1_reshaped = parent1.reshape(-1, 3)
            parent2_reshaped = parent2.reshape(-1, 3)

            offspring1 = np.concatenate(
                (parent1_reshaped[:crossover_point], parent2_reshaped[crossover_point:])
            )
            offspring2 = np.concatenate(
                (parent2_reshaped[:crossover_point], parent1_reshaped[crossover_point:])
            )

            offspring1 = offspring1.reshape(*self.config.chromosome_dim, 3)
            offspring2 = offspring2.reshape(*self.config.chromosome_dim, 3)

            return offspring1, offspring2

        return parent1.copy(), parent2.copy()

    def _mutate(self, offspring: np.ndarray) -> np.ndarray:
        """Mutate a single RGB value in the offspring's chromosome using Gaussian noise.

        With probability p_mutation, selects one random position in the grid and applies
        Gaussian noise to its RGB values. The resulting values are clipped to valid
        RGB range [0, 255].

        Args:
            offspring: Chromosome to mutate, shape (height, width, 3)

        Returns:
            np.ndarray: Mutated chromosome of shape (height, width, 3). If no mutation
                occurs, returns the original chromosome unchanged.
        """
        if np.random.rand() < self.config.p_mutation:
            index_to_mutate = np.random.randint(0, offspring.reshape(-1, 3).shape[0])

            # shape: (3,)
            sub_tile_to_mutate = offspring.reshape(-1, 3)[index_to_mutate]

            for channel in range(sub_tile_to_mutate.shape[-1]):
                noise = np.random.normal(0, self.config.mutation_std)
                sub_tile_to_mutate[channel] = np.clip(
                    sub_tile_to_mutate[channel] + noise, 0, 255
                )

            offspring.reshape(-1, 3)[index_to_mutate] = sub_tile_to_mutate

            offspring = offspring.reshape(*self.config.chromosome_dim, 3)

        return offspring

    def _should_stop(
        self, elite_fitness: float, generations_without_improvement: int
    ) -> bool:
        """Determine if the evolution should stop.

        Checks two stopping conditions:
        1. Perfect solution found (fitness = 0)
        2. No improvement for specified number of generations

        Args:
            elite_fitness: Fitness value of the best individual
            generations_without_improvement: Number of consecutive generations without
                fitness improvement

        Returns:
            bool: True if either stopping condition is met, False otherwise
        """
        return (
            elite_fitness == 0
            or generations_without_improvement > self.config.early_stopping_patience
        )

    def evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """Run the genetic algorithm evolution process.

        Executes the main evolutionary loop:
        1. Evaluates current population
        2. Updates statistics and checks stopping conditions
        3. Creates new population through selection, crossover, and mutation
        4. Preserves elite individual

        The process continues until either:
        - Maximum generations reached
        - Perfect solution found (fitness = 0)
        - No improvement for specified number of generations

        Returns:
            tuple containing:
                - np.ndarray: Array of shape (n_generations, height, width, 3) containing
                    best chromosomes from each generation
                - np.ndarray: Array of shape (n_generations,) containing best fitness
                    values from each generation
        """

        for generation in range(self.config.max_gen):
            self.current_generation = generation

            # Evaluate current population
            fitness: np.ndarray = self._calc_fitness(self.population)
            elite, elite_fitness = self._update_statistics(fitness)

            # early stopping check
            if elite_fitness < self.best_fitness:
                self.best_fitness = elite_fitness
                self.generation_without_improvement = 0
            else:
                self.generation_without_improvement += 1

            if self._should_stop(elite_fitness, self.generation_without_improvement):
                print(f"Early stopping at generation {generation}")
                break

            if generation % 100 == 0:
                print(f"Generation {generation} \t best fitness: {elite_fitness:.5f}")

            if generation == self.config.max_gen - 1:
                print(f"Reached max generation: {generation+1}")

            # Create new population
            mating_pool = self._generate_mating_pool(fitness)
            new_population = self._create_offspring(mating_pool)

            # Update population
            self.population = np.vstack((new_population, np.expand_dims(elite, 0)))

        return np.array(self.best_chromosomes), np.array(self.best_fitness_values)


class RealTimeVisualizer:
    """Visualization tools for genetic algorithm results.

    This class provides static methods for visualizing the evolution
    process through fitness plots and color animations.
    """

    def __init__(self, target_color: np.ndarray, chromosome_dim: tuple[int, int]):
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.target_color = target_color
        self.chromosome_dim = chromosome_dim
        (self.fitness_line,) = self.ax1.plot([], [])
        self.generation_text = self.ax1.text(
            0.02, 0.95, "", transform=self.ax1.transAxes
        )

        # Setup fitness plot
        self.ax1.set_title("Fitness Over Time")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness")
        self.ax1.grid(True)

        # Setup color grid display
        self.ax2.set_title("Best Individual vs Target")
        self.img = None

    def update(
        self,
        generation: int,
        best_fitness: float,
        fitness_history: list[float],
        best_chromosome: np.ndarray,
    ):
        # Update fitness plot
        self.fitness_line.set_data(range(len(fitness_history)), fitness_history)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update generation text
        self.generation_text.set_text(
            f"Generation: {generation}\nFitness: {best_fitness:.2f}"
        )

        # Update color grid
        if self.img is None:
            grid = self._create_comparison_grid(best_chromosome)
            self.img = self.ax2.imshow(grid)
        else:
            self.img.set_array(self._create_comparison_grid(best_chromosome))

        plt.pause(0.01)

    def _create_comparison_grid(self, chromosome: np.ndarray) -> np.ndarray:
        # Create a grid showing target color and best chromosome side by side
        h, w = self.chromosome_dim
        grid = np.zeros((h, w * 2 + 1, 3))

        # Left side: current best chromosome
        grid[:, :w] = chromosome / 255.0

        # Middle: separator
        grid[:, w] = 0.5

        # Right side: target color
        grid[:, w + 1 :] = self.target_color / 255.0

        return grid

    def save(self, filename: str):
        plt.savefig(filename)

    @staticmethod
    def plot_fitness_history(fitness_history: np.ndarray) -> None:
        """Plot the fitness history over generations.

        Args:
            fitness_history: Array of best fitness values per generation
        """
        plt.figure(figsize=(10, 5))
        plt.plot(fitness_history)
        plt.title("Best Chromosome Fitness in Each Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid()
        plt.show()

    @staticmethod
    def create_animation(
        target_color: np.ndarray,
        chromosomes: np.ndarray,
        chromosome_dim: int,
        save_path: str | None = None,
    ) -> None:
        """Create an animation showing the evolution of colors.

        Args:
            target_color: Target RGB color
            chromosomes: Array of best chromosomes per generation
            save_path: Optional path to save the animation
        """
        # Plot the color of the best chromosome in a generation
        fig = plt.figure(figsize=(chromosome_dim + 2, chromosome_dim + 2))

        def create_grid(chromosome):
            grid = np.zeros((chromosome_dim + 2, chromosome_dim + 2, 3))
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = target_color / 255.0
            grid[1:-1, 1:-1] = chromosome / 255.0
            return grid

        def update(frame):
            img.set_array(create_grid(chromosomes[frame]))
            title.set_text(f"Generation {frame}")
            return img, title

        # Setup initial frame
        img = plt.imshow(create_grid(chromosomes[0]))
        title = plt.title("Generation 0")

        anim = FuncAnimation(
            fig,
            func=update,
            frames=range(len(chromosomes)),
            interval=20,
            blit=True,
        )
        if save_path:
            anim.save(save_path, writer="pillow")

        plt.close()


def main() -> None:

    config = GAConfig(
        popsize=1000,
        max_gen=1000,
        p_crossover=0.8,
        p_mutation=0.01,
        chromosome_dim=(4, 4),
        target_value=np.array([222, 165, 33]),
    )
    print(f"TARGET_VALUE: {config.target_value}")

    # Run GA
    ga = ColorMatchingGA(config)
    best_chromosomes, best_fitness_values = ga.evolve()

    print(f"Target value: {config.target_value}")
    # Visualize results
    visualizer = RealTimeVisualizer(config.target_value, config.chromosome_dim)
    visualizer.plot_fitness_history(best_fitness_values)
    visualizer.create_animation(
        config.target_value,
        best_chromosomes,
        config.chromosome_dim[0],
        save_path=f"animation_{datetime.now().strftime('%Y%m%d_%H%M')}.gif",
    )


if __name__ == "__main__":
    main()
