import random
import numpy as np
from sklearn.base import accuracy_score
from sklearn.model_selection import train_test_split


class Genetic_Alogorithm:
    def __init__(self, X, y, num_params, num_solutions=100, mutation_rate=0.2):
        self.X = X
        self.y = y
        self.num_params = num_params
        self.num_solutions = num_solutions
        self.mutation_rate = mutation_rate
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def initialize_solutions(self):
        solutions = []
        for _ in range(self.num_solutions):
            # Sử dụng np.random.uniform để khởi tạo giá trị trong khoảng từ -500 đến 500
            solution = np.random.uniform(-500, 500, size=self.num_params)
            solutions.append(tuple(solution))
        return solutions

    def mutate(self, child, mutation_rate):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        mutated_child = tuple(
            [
                gene * np.random.uniform(1 - self.mutation_rate, 1 + self.mutation_rate)
                for gene in child
            ]
        )
        return mutated_child

    def crossover(self, parent1, parent2, type=None):
        if type is None or type =="one_point":
            crossover_point = np.random.randint(1, len(parent1) - 1)
            child = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
            return tuple(child)
        elif type == "two_point":
            crossover_point1, crossover_point2 = np.sort(
                np.random.choice(range(1, len(parent1)), 2, replace=False)
)
            child = (
                list(parent1[:crossover_point1])
                + list(parent2[crossover_point1:crossover_point2])
                + list(parent1[crossover_point2:])
            )
            return tuple(child)
        elif type == "mean":
            child = [(parent1[i] + parent2[i]) / 2 for i in range(len(parent1))]
            return tuple(child)

    def evolve(self, solutions, crossover = "None", mutate_rate = "None"):
        rankedsolutions = [(self.fitness(theta), theta) for theta in solutions]
        rankedsolutions = sorted(rankedsolutions, key=lambda x: x[0], reverse=True)
        y_pred = self.predict(self.X_train, solutions[0])
        print(f'fitness:{rankedsolutions[0][0]}, accuracy: {accuracy_score(self.y_train, y_pred)}')
        bestSolutions = rankedsolutions[:20] + rankedsolutions[-5:]

        new_solution = [rankedsolutions[0][1]]

        for _ in range(self.num_solutions - 1):
            parent1, parent2 = (
                random.choice(bestSolutions)[1],
                random.choice(bestSolutions)[1],
            )
            child1 = self.crossover(parent1, parent2, crossover)
            child1 = self.mutate(child1, mutate_rate)
            new_solution.append(child1)
        return np.array(new_solution), rankedsolutions[0][0]

    # support funcion
    def fitness(self, theta):
        result = 0
        y_pred = self.predict(self.X_train, theta)
        return accuracy_score(self.y_train, y_pred)
        for yp, yt in zip(y_pred, self.y_train):
            diff = abs(yt - yp)
            if diff == 0:
                continue
            elif diff == 1:
                result += 200
            elif diff == 2:
                result += 400
            elif diff == 3:
                result += 800
            else:
                result += 2000
        return result
    def predict(self, matrices, theta):
        list_scores = np.dot(matrices, theta)
        label = []
        for score in list_scores:
            if  score < 580:
                label.append(0)
            elif score >=580 and score < 670:
                label.append(1)
            elif score >= 670 and score < 740:
                label.append(2)
            elif score >= 740 and score < 800:
                label.append(3)
            elif score >= 800:
                label.append(4)
        return np.array(label)
    