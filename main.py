from __future__ import annotations

import sys
from enum import Enum
from copy import copy, deepcopy
import random
from typing import List

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
    _strings: List[str]

    fitness: int
    words: List[Word]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self._n = n
        self._m = m

        self._strings = strings
        self.calculate_fitness()

        self.words = self._generate_random_positions()

    def __str__(self):
        return ";".join([str(x) for x in self.words])

    def __copy__(self) -> Crossword:
        crossword = Crossword(self.strings, self.n, self.m)
        crossword.words = [copy(word) for word in self.words]
        return crossword

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def strings(self) -> List[str]:
        return self._strings

    def _generate_random_positions(self) -> List[Word]:
        words = []
        for word in self.strings:
            direction = random.choice(list(Direction))
            constraint_x = self.n - 1 - (len(word) if direction == Direction.HORIZONTAL else 0)
            constraint_y = self.m - 1 - (len(word) if direction == Direction.VERTICAL else 0)

            words.append(Word(
                x=random.randint(0, constraint_x),
                y=random.randint(0, constraint_y),
                direction=direction,
                value=word
            ))

        return words

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

            self._selection()

        self.calculate_fitness()

        self.calculate_fitness()

        best_individual = self.find_best_individual()

        current_best_fitness = best_individual.fitness
        print(f"Best fitness: {current_best_fitness}")
        best_individual.print()

    def _selection(self, population: List[Crossword]) -> List[Crossword]:
        pass

    def _crossover(self, parent1: Crossword, parent2: Crossword, crossover_rate: float = 0.5) -> Crossword:
        child = copy(parent1)

        for i in range(len(parent1.words)):
            if random.random() < crossover_rate:
                child.words[i].x = parent2.words[i].x
                child.words[i].y = parent2.words[i].y
                child.words[i].direction = parent2.words[i].direction

        return child

    def _mutation(self, population: List[Crossword], mutation_rate: float = 0.5):
        pass


def main():
    crossword = EvolutionaryAlgorithm(population_size=500,
                                      strings=["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"])
    crossword.run()


if __name__ == "__main__":
    main()
