"""
Author: Enrique Vilchez Campillejo
"""

import random

from jmetal.core.solution import PermutationSolution
from jmetal.operator.mutation import PermutationSwapMutation


class PermutationSwapMutationModified(PermutationSwapMutation):

    def __init__(self, probability: float):
        super().__init__(probability=probability)

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        rand = random.random()

        if rand <= self.probability:
            positions = random.sample(range(solution.number_of_variables - 1), 20)
            it = iter(positions)
            for pos_one in it:  # make sure it has pair elements, otherwise throws StopIteration exception
                pos_two = next(it)
                solution.variables[pos_one], solution.variables[pos_two] = \
                    solution.variables[pos_two], solution.variables[pos_one]

        return solution
