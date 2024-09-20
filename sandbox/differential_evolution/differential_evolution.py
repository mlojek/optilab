'''
Basic DE/current-to-pbest/1 algorithm
'''
import random
from typing import List, Tuple


class DifferentialEvolution:
    def __init__(self) -> None:
        pass

    def optimize(
        self,
        function: callable,
        call_budget: int,
        population_size: int,
        target_value: float,
        allowed_error: float,
        dimensions: int,
        p: int,
        mutation_factor: float,
        crossover_rate: float,
        bounds: Tuple[float, float]=(-100, 100)
    ) -> float:
        '''
        Perfom optimization of a function
        '''
        # parameters of the DE
        self.dimensions = dimensions
        self.population_size = population_size
        self.target_value = target_value
        self.allowed_error = allowed_error
        self.p = p
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.bounds = bounds

        # population and evaluations
        self.population = []
        self.evaluations = []
        self.new_population = []
        self.new_evaluations = []
        self.best_value = None

        # fitness function
        self.function = function
        self.calls_left = call_budget

        # optimization process itself
        self.initialize_population()
        self.evaluate_population()
        while not self.is_end():
            self.mutate()
            self.crossover()
            self.evaluate_new_population()
            self.selection()
        
        return self.best_value
    
    def is_end(self) -> bool:
        '''
        Check if the optimization process should be ended. This can happen
        either when the call budget has been expended or when a solution of
        sufficient quality has been reached.

        :return: true if the optimization process should be ended, false otherwise
        '''
        return self.calls_left == 0 or self.best_value - self.target_value < self.allowed_error
    
    def evaluate_specimen(self, specimen: List[float]) -> float:
        '''
        Evaluate a single speciment and return it's fitness value

        :param specimen: vector to evaluate
        :return: fitness value of provided vector
        '''
        if self.calls_left > 0:
            self.calls_left -= 1
            return self.function(specimen)
        else:
            return None
    
    def initialize_population(self) -> None:
        '''
        Creates a random starting population.
        '''
        self.population = [
            [
                random.uniform(*self.bounds)
                for _ in self.dimensions
            ]
            for _ in self.population_size
        ]

    def evaluate_population(self) -> None:
        '''
        Calculate fitness values for the current population.
        '''
        self.evaluations = [
            self.evaluate_specimen(vec)
            for vec in self.population
        ]

    def evaluate_new_population(self) -> None:
        '''
        Calculate fitness values for the new population.
        '''
        self.new_evaluations = [
            self.evaluate_specimen(vec)
            for vec in self.new_population
        ]

    def selection(self) -> None:
        '''
        Compares the current population and new population and appends better
        mutants to current population.
        '''
        for curr_vec, curr_val, new_vec, new_val in zip(self.population, self.evaluations, self.new_population, self.new_evaluations):
            if new_val:
                if new_val > curr_val:
                    curr_vec = new_vec

    # TODO OOP, bounds
    def mutation(self, population: List[List[float]], evaluations: List[float], p: int=1, mutation_factor: float=1.0) -> List[List[float]]:
        '''
        TODO
        '''
        sorted_evaluations, sorted_population = zip(*sorted(zip(evaluations, population)))
        p_best = sorted_population[:p]

        mutated_population = []
        for specimen in population:
            # TODO element-wise operations
            current_to_best = random.choice(p_best) - specimen
            random_to_random = random.choice(population) - random.choice(population)
            new_specimen = specimen + mutated_population * (current_to_best + random_to_random)
            mutated_population.append(new_specimen)

        return mutated_population

