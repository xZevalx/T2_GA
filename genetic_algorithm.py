import numpy as np
import random as pyrandom


class IOrganism:
    """
    Base organism class (interface). Subclasses should implement:
    - spontaneous_generation(size): Useful when initializing a new population
    - mutate(*args, **kwargs)
    - breed(organism)
    """

    def __init__(self, representation=None, size=0):
        """
        Organism constructor.

        :param representation: List with elements that represent its genome.
        If None applies 'spontaneous generation'
        """
        if representation is None:
            self.spontaneous_generation(size)
        else:
            self.representation = representation if not isinstance(representation, np.ndarray) else np.array(
                representation)

        self.size = len(self.representation)

        # Variable to store organism's fitness
        self._fitness = 0

    def spontaneous_generation(self, size):
        """
        Creates an organism's representation from nothing with 'size' genes.
        Should assign the member self.representation
        """
        pass

    def mutate(self, *args, **kwargs):
        """ Applies an inplace mutation to self.representation """
        pass

    def breed(self, organism):
        """
        Breed a new organism based on self and organism.
        Return a new organism instance
        """
        pass

    def set_fitness(self, f):
        self._fitness = f

    def get_fitness(self):
        return self._fitness

    fitness = property(get_fitness, set_fitness)


class GeneticAlg:
    def __init__(self,
                 pop_size,
                 organism_type,
                 organism_size,
                 fitness_function,
                 selection_function,
                 mutation_rate=.1,
                 organism_kwargs={}):
        """

        :param pop_size: Size of the population
        :param organism_type: A subclass of IOrganism
        :param organism_size: Length of organism representation
        :param fitness_function: function that takes only the organism to calculate its fitness
        :param selection_function: function that selects the population to give breed.
                                   Signature sf(organism, fitnesses, return_best)
        :param organism_kwargs: Extra keyword arguments for organism_type. Implementation dependant
        :param mutation_rate: float
        """
        self.population_size = pop_size
        # Init population
        self.population = np.array([organism_type(size=organism_size, **organism_kwargs) for _ in range(pop_size)])
        self.fitness_function = fitness_function
        self.mut_rate = mutation_rate
        self.selection_function = selection_function
        self.fitness = np.array([0 for _ in range(pop_size)])

    def calc_fitness(self):
        fitness = []
        for organism in self.population:
            organism.fitness = self.fitness_function(organism)
            fitness.append(organism.fitness)

        return np.array(fitness)

    def offspring_generation(self, elite_organisms, n_offspring):
        """ Randomly selects parents to generate offspring. Also performs mutation """
        offsprings = []
        elite_organisms = elite_organisms.tolist()
        for _ in range(n_offspring):
            parent1, parent2 = pyrandom.sample(elite_organisms, 2)
            offsprings.append(parent1.breed(parent2))
            if np.random.random() < self.mut_rate:
                offsprings[-1].mutate()
        return offsprings

    def evolve(self, generations=10, trace_evolution=False):
        """
        Evolution process

        :param generations: Number of generations
        :param trace_evolution: Boolean. To trace evolution the given selection_function must return the best organism
                                as a second value if is requested
        :return: (int) generations processed is trace_evolution is False. Else generation processed plus tuple of best
                 organism per generation and its fitness

        """

        assert generations > 0

        fitness_evolution = []
        organism_evolution = []
        for i in range(generations):
            print('Evaluating generation {}'.format(i+1))
            # Evaluate population fitness
            self.fitness = self.calc_fitness()
            # Select population to give breed
            elite_organisms, best_organism = self.selection_function(self.population, self.fitness, True)
            if trace_evolution:
                organism_evolution.append(best_organism)
                fitness_evolution.append(best_organism.fitness)

            # Create new offsprings
            offsprings = self.offspring_generation(elite_organisms, self.population_size - len(elite_organisms))
            # Update population
            self.population = np.concatenate((elite_organisms, offsprings))

        print('Evolution process finished with {} generations'.format(i+1))

        if trace_evolution:
            return i, organism_evolution, fitness_evolution
        else:
            return i
