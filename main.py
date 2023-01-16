"""
Author: Enrique Vilchez Campillejo
"""

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

from problem import ImagesProblem


def evolutionary_algorithm():
    problem = ImagesProblem()

    algorithm = GeneticAlgorithm(
        problem=problem,
        crossover=PMXCrossover(probability=1.0),
        selection=BinaryTournamentSelection(),
        mutation=PermutationSwapMutation(probability=0.3),
        population_size=20,
        offspring_population_size=2,
        termination_criterion=StoppingByEvaluations(max_evaluations=100)
    )

    start = time.time()
    algorithm.run()
    end = time.time()
    total = end - start

    solution = algorithm.get_result()
    print('Best fitness (minimum):', solution.objectives[0])
    print('Ftiness mean: ' + str(np.mean(problem.fitness_history)))
    print('Ftiness median: ' + str(np.median(problem.fitness_history)))
    print('Ftiness std: ' + str(np.std(problem.fitness_history)))
    print('GA spent time: ' + str(total))
    print('Best solution:', solution.variables)

    plt.plot(problem.fitness_history)
    plt.title('GA-ImagesProblem | Performance')
    plt.xlabel('Iterations')
    plt.ylabel('Average fitness')
    plt.show()
    return solution, problem.images_matrix


def show_image(images_matrix):
    sb.heatmap(images_matrix)
    plt.show()


def obtain_matrix_from_solution(solution, images_matrix):
    permutation = solution.variables
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    images_matrix[:] = images_matrix[idx, :]
    return images_matrix


if __name__ == '__main__':
    solution, images_matrix = evolutionary_algorithm()
    show_image(obtain_matrix_from_solution(solution, images_matrix))
