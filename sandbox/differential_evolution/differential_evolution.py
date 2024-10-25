'''
Basic DE/current-to-pbest/1 algorithm
'''
import random
from typing import List, Tuple
from cec2017.functions import f1


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
        TODO parameters
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
        if self.calls_left == 0:
            return True
        elif self.best_value:
            return self.best_value - self.target_value < self.allowed_error
        return False
            
    
    def evaluate_specimen(self, specimen: List[float]) -> float:
        '''
        Evaluate a single speciment and return it's fitness value

        :param specimen: vector to evaluate
        :return: fitness value of provided vector
        '''
        if self.calls_left > 0:
            self.calls_left -= 1
            return self.function([specimen])[0]
        else:
            return None
    
    def initialize_population(self) -> None:
        '''
        Creates a random starting population.
        '''
        self.population = [
            [
                random.uniform(*self.bounds)
                for _ in range(self.dimensions)
            ]
            for _ in range(self.population_size)
        ]

    def evaluate_population(self) -> None:
        '''
        Calculate fitness values for the current population.
        '''
        self.evaluations = [
            self.evaluate_specimen(vec)
            for vec in self.population
        ]
        if self.best_value:
            self.best_value = min(self.best_value, min(self.evaluations))
        else:
            self.best_value = min(self.evaluations)

    def evaluate_new_population(self) -> None:
        '''
        Calculate fitness values for the new population.
        '''
        self.new_evaluations = [
            self.evaluate_specimen(vec)
            for vec in self.new_population
        ]
        self.best_value = min(self.best_value, min(self.new_evaluations))

    def selection(self) -> None:
        '''
        Compares the current population and new population and appends better
        mutants to current population.
        '''
        final_population = []
        for curr_vec, curr_val, new_vec, new_val in zip(self.population, self.evaluations, self.new_population, self.new_evaluations):
            if new_val:
                if new_val < curr_val:
                    final_population.append(new_vec)
                else:
                    final_population.append(curr_vec)
        self.population = final_population

    def mutate(self) -> None:
        '''
        TODO
        '''
        _, sorted_population = zip(*sorted(zip(self.evaluations, self.population)))
        p_best = sorted_population[:self.p]

        mutated_population = []
        for specimen in self.population:
            current_to_best = [a - b for a, b in zip(random.choice(p_best), specimen)]
            random_to_random = [a - b for a, b in zip(random.choice(self.population), random.choice(self.population))]
            new_specimen = [a + self.mutation_factor * (b + c) for a, b, c in zip(specimen, current_to_best, random_to_random)]
            # TODO check for bounds
            mutated_population.append(new_specimen)

        self.new_population = mutated_population

    def crossover(self) -> None:
        pass


if __name__ == '__main__':
    de = DifferentialEvolution()
    optimal_solution = de.optimize(
        f1,
        10 * 1e4,
        4 * 10,
        100,
        1e-8,
        10,
        5,
        0.001,
        0.1
    )
    print(optimal_solution)
    print(de.calls_left)