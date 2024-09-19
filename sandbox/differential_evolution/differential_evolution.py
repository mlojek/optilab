'''
Basic DE/current-to-pbest/1 algorithm
'''
import random
from typing import List


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
        lower_bound: float,
        upper_bound: float,
        p: int,
        mutation_factor: float,
        crossover_rate: float,
    ) -> float:
        '''
        Perfom optimization of a function
        '''
        # initialize population
        # in loop while best value not close enough or call budget available
            # mutation
            # crossover
            # selection
        # return best value
    
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

