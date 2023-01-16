"""
Author: Enrique Vilchez Campillejo
"""

import random

from jmetal.core.solution import PermutationSolution
from jmetal.operator.mutation import ScrambleMutation


class ScrambleMutationModified(ScrambleMutation):

    def __init__(self, probability: float):
        super().__init__(probability=probability)

    def execute(self, solution: PermutationSolution) -> PermutationSolution:
        rand = random.random()

        if rand <= self.probability:
            # Here is a modification, from 2D list to 1D for the specific problem
            point1 = random.randint(0, len(solution.variables))
            point2 = random.randint(0, len(solution.variables) - 1)

            if point2 >= point1:
                point2 += 1
            else:
                point1, point2 = point2, point1

            if point2 - point1 >= 20:
                point2 = point1 + 20

            # Here is a modification, from 2D list to 1D for the specific problem
            values = solution.variables[point1:point2]
            solution.variables[point1:point2] = random.sample(values, len(values))

        return solution
