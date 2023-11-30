from __future__ import annotations

import sys
from enum import Enum
from copy import copy, deepcopy
import random
from typing import List

Y_POSITION_MAX = 20
X_POSITION_MAX = 20

MAX_INT = sys.maxsize


class Direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class GraphCell(Enum):
    EMPTY = 0
    FILLED = 1


class Graph:
    _matrix: List[List[GraphCell]]
    _n: int

    def __init__(self, n: int):
        self._n = n
        self._matrix = [[GraphCell.EMPTY for _ in range(n)] for _ in range(n)]

    def add_edge(self, u: int, v: int) -> None:
        self._matrix[u][v] = GraphCell.FILLED
        self._matrix[v][u] = GraphCell.FILLED

    def is_connected(self) -> bool:
        return all(self._dfs())

    def number_of_disconnected_nodes(self) -> int:
        return len(list(filter(lambda x: not x, self._dfs())))

    def _dfs(self, i=0, visited=None) -> List[bool]:
        if visited is None:
            visited = [False for _ in range(self._n)]

        visited[i] = True
        for j in range(self._n):
            if self._matrix[i][j] == GraphCell.FILLED and not visited[j]:
                self._dfs(j, visited)

        return visited


class Word:
    _value: str

    x: int
    y: int
    direction: Direction

    def __init__(self, value: str, x: int, y: int, direction: Direction) -> None:
        self._value = value

        self.x = x
        self.y = y
        self.direction = direction

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.direction.value}"

    def __copy__(self) -> Word:
        return Word(self.value, self.x, self.y, self.direction)

    @property
    def value(self) -> str:
        return self._value

    @property
    def length(self) -> int:
        return len(self._value)


class Crossword:
    _n: int
    _m: int

    words: List[Word]
    fitness: int

    def __init__(self, genes, n: int = 20, m: int = 20):
        self._n = n
        self._m = m

        self.words = genes
        self.calculate_fitness()

    def __str__(self):
        return ";".join([str(x) for x in self.words])

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    def within_bounds(self, x: int, y: int) -> bool:
        return (0 <= x < self.n) and (0 <= y < self.m)

    def word_within_bounds(self, word: Word) -> bool:
        end_x = word.x + (word.length if word.direction == Direction.HORIZONTAL else 0)
        end_y = word.y + (word.length if word.direction == Direction.VERTICAL else 0)

        return self.within_bounds(word.x, word.y) and self.within_bounds(end_x, end_y)

    def print(self) -> None:
        matrix = [[" . " for _ in range(0, self.m)] for _ in range(0, self.n)]

        for word in self.words:
            for i in range(word.length):
                new_x = word.x + (i if word.direction == Direction.HORIZONTAL else 0)
                new_y = word.y + (i if word.direction == Direction.VERTICAL else 0)
                if self.within_bounds(new_x, new_y):
                    matrix[new_x][new_y] = f" {word.value[i]} "

        print(" " + " - " * self.m + " ")
        for row in matrix:
            print(f'|{"".join(row)}|')
        print(" " + " - " * self.m + " ")

    def calculate_fitness(self) -> None:
        pass


class EvolutionaryAlgorithm:
    _strings: List[str]
    _population_size: int

    population: List[Crossword]

    def __init__(self, strings: List[str], population_size: int = 500):
        self._strings = strings
        self._population_size = population_size

        self.population = []

    @property
    def strings(self) -> List[str]:
        return self._strings

    @property
    def population_size(self) -> int:
        return self._population_size

    def calculate_fitness(self) -> None:
        for crossword in self.population:
            crossword.calculate_fitness()

    def find_best_individual(self) -> Crossword:
        return max(self.population, key=lambda x: x.fitness)

    def run(self, max_generation=20000):
        best_fitness = MAX_INT
        stagnant_generations = 0

        for generation in range(max_generation):
            print(f"Generation {generation}")
            self.calculate_fitness()

            best_individual = self.find_best_individual()

            current_best_fitness = best_individual.fitness
            print(f"Best fitness: {current_best_fitness}")
            best_individual.print()

            if current_best_fitness == 0:
                break

            if current_best_fitness == best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = current_best_fitness
                stagnant_generations = 0

            self.__selection()

        self.calculate_fitness()

        self.calculate_fitness()

        best_individual = self.find_best_individual()

        current_best_fitness = best_individual.fitness
        print(f"Best fitness: {current_best_fitness}")
        best_individual.print()

    def __selection(self):
        pass

    def __crossover(self, parent1, parent2):
        pass

    def __mutation(self, selection, mutation_rate):
        pass


def main():
    crossword = EvolutionaryAlgorithm(population_size=500,
                                      strings=["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"])
    crossword.run()


if __name__ == "__main__":
    main()
