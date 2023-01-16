"""
Author: Enrique Vilchez Campillejo
"""

from abc import ABC
import numpy as np
import time

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution


class ImagesProblem(PermutationProblem, ABC):

    def __init__(self):
        super().__init__()

        images_matrix, number_of_pixels = self.read_from_file('img1.txt')

        self.images_matrix = images_matrix
        self.fitness_history = []

        self.obj_directions = [self.MINIMIZE]
        self.number_of_variables = number_of_pixels
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.number_of_splits = 1

        self.subtractions = {}
        self.subtract_rows()

    def read_from_file(self, instance: str):
        matrix = np.loadtxt(instance, dtype=int)
        return matrix, len(matrix)

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
        # new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)
        new_solution.variables = [i for i in
                                  range(0, self.number_of_variables)]  # take initial order as initial solution

        if 1 < self.number_of_splits < 512:
            self.number_of_splits *= 2
            chunked_variables = self.chunk_it(new_solution.variables, self.number_of_splits)
            random_permutation = np.random.permutation(chunked_variables)
            new_solution.variables = random_permutation.flatten().tolist()
        else:
            self.number_of_splits = 2

        return new_solution

    def chunk_it(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness = []
        for index, row1 in enumerate(solution.variables):
            if index == len(solution.variables) - 1:
                break

            row2 = solution.variables[index + 1]
            if row1 < row2:
                search = str(row1) + '_' + str(row2)
            else:
                search = str(row2) + '_' + str(row1)
            subtraction = self.subtractions[search]
            fitness.append(np.sum(subtraction))

        solution.objectives[0] = sum(fitness) / len(fitness)
        self.fitness_history.append(solution.objectives[0])
        #print(solution)
        return solution

    def subtract_rows(self):
        start = time.time()
        for i in range(0, self.number_of_variables - 1):
            for j in range(i + 1, self.number_of_variables):
                self.subtractions[str(i) + "_" + str(j)] = \
                    np.sum(np.absolute(self.images_matrix[i][:]
                                       - self.images_matrix[j][:]))
        end = time.time()
        total = end - start
        print('Subtractions spent time: ' + str(total))

    def get_name(self) -> str:
        return 'ImagesProblem'
