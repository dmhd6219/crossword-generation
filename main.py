from __future__ import annotations

import os
import sys
from enum import Enum
from copy import copy
import random
from typing import List, Set

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

    def fill_edge(self, u: int, v: int) -> None:
        self._matrix[u][v] = GraphCell.FILLED
        self._matrix[v][u] = GraphCell.FILLED

    def is_connected(self) -> bool:
        return self.get_amount_of_disconnected() == 0

    def get_amount_of_disconnected(self) -> int:
        return self._n - len(self._dfs())

    def _dfs(self, current: int = 0, visited: Set[int] = None) -> Set[int]:
        if visited is None:
            visited = set()

        visited.add(current)
        for new in range(self._n):
            if self._matrix[current][new] == GraphCell.FILLED and new not in visited:
                self._dfs(new, visited)

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
        self._strings = strings
        self._n = n
        self._m = m

        self._strings = strings

        self.words = self._generate_random_positions()
        self.fitness = self.calculate_fitness()

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

    def generate_safe_x_from_word(self, word: Word) -> int:
        constraint = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
        return random.randint(0, constraint)

    def generate_safe_y_from_word(self, word: Word) -> int:
        constraint = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)
        return random.randint(0, constraint)

    def generate_safe_x_from_string(self, word: str, direction: Direction) -> int:
        constraint = self.n - 1 - (len(word) if direction == Direction.HORIZONTAL else 0)
        return random.randint(0, constraint)

    def generate_safe_y_from_string(self, word: str, direction: Direction) -> int:
        constraint = self.m - 1 - (len(word) if direction == Direction.VERTICAL else 0)
        return random.randint(0, constraint)

    def _generate_random_positions(self) -> List[Word]:
        words = []
        for word in self.strings:
            direction = random.choice(list(Direction))

            words.append(Word(
                x=self.generate_safe_x_from_string(word, direction),
                y=self.generate_safe_y_from_string(word, direction),
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
        print("\n")

    def detect_overlapping(self, word1: Word, word2: Word) -> bool:
        if word1.direction != word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            if not (abs(word1.x - word2.x) <= 1):
                return False

            if word2.y < word1.y:
                word1, word2 = word2, word1

            return word1.y <= word2.y <= word1.y + word1.length

        elif word1.direction == Direction.HORIZONTAL:
            if not abs(word1.y - word2.y) <= 1:
                return False

            if word2.x < word1.x:
                word1, word2 = word2, word1

            return word1.x <= word2.x <= word1.x + word1.length

        return False

    def check_intersection(self, word1: Word, word2: Word) -> bool:
        if word1.direction == word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            return (word2.x <= word1.x < word2.x + len(word2.value) and
                    word1.y <= word2.y < word1.y + len(word1.value))
        else:
            return (word1.x <= word2.x < word1.x + len(word1.value) and
                    word2.y <= word1.y < word2.y + len(word2.value))

    def check_letter_match(self, word1: Word, word2: Word) -> bool:
        if word1.direction == word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            return word1.value[word2.y - word1.y] == word2.value[word1.x - word2.x]
        else:
            return word1.value[word2.x - word1.x] == word2.value[word1.y - word2.y]

    def check_collisions(self, word1: Word, word2: Word) -> int:
        if word1.direction == word2.direction:
            return 0

        collisions = 0

        if word1.direction == Direction.HORIZONTAL:
            word1, word2 = word2, word1

        if word2.y == word1.y - 1 and word2.x <= word1.x < word2.x + len(word2.value):
            collisions += 1

        if word2.y == word1.y + len(word1.value) and word2.x <= word1.x < word2.x + len(word2.value):
            collisions += 1

        if word2.x + len(word2.value) - 1 == word1.x - 1:
            if word1.y <= word2.y <= word1.y + len(word1.value) - 1:
                collisions += 1

        if word2.x == word1.x + 1:
            if word1.y <= word2.y <= word1.y + len(word1.value) - 1:
                collisions += 1

        return collisions

    def calculate_fitness(self) -> int:
        graph = Graph(len(self.words))
        fitness = 0

        for i in range(len(self.words)):
            word1 = self.words[i]

            fitness += 10000 * (not self.word_within_bounds(word1))

            for j in range(i + 1, len(self.words)):
                word2 = self.words[j]

                if word1.direction == word2.direction:
                    fitness += 10 * self.detect_overlapping(word1, word2)
                else:
                    if self.check_intersection(word1, word2):
                        graph.fill_edge(i, j)
                        graph.fill_edge(j, i)
                        fitness += 3 * (not self.check_letter_match(word1, word2))

                    fitness += self.check_collisions(word1, word2)

        fitness += graph.get_amount_of_disconnected() * 100
        return fitness


class EvolutionaryAlgorithm:
    _strings: List[str]

    _n: int
    _m: int
    _population_size: int

    population: List[Crossword]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20, population_size: int = 100):
        self._strings = strings
        self._n = n
        self._m = m
        self._population_size = population_size

        self.population = []

    @property
    def strings(self) -> List[str]:
        return self._strings

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    def calculate_fitnesses(self) -> None:
        for crossword in self.population:
            crossword.fitness = crossword.calculate_fitness()

    def generate_random_population(self, population_size=300):
        return [Crossword(self.strings, self.m, self.n) for _ in range(population_size)]

    def _selection(self, initial_population: List[Crossword], best_individuals_percentage=0.3):
        population = [copy(crossword) for crossword in initial_population]

        best_individuals_length = int(len(population) * best_individuals_percentage)
        rest_individuals_length = len(population) - best_individuals_length

        best_individuals = population[:best_individuals_length:]
        rest_individuals = random.sample(population, rest_individuals_length)

        new_individuals = [
            self._crossover(
                self._tournament_selection(rest_individuals),
                self._tournament_selection(rest_individuals)
            )
            for _ in range(len(rest_individuals))
        ]

        return best_individuals + self._mutate_population(new_individuals)

    def _tournament_selection(self, population: List[Crossword], tournament_size=3):
        return min(random.sample(population, k=tournament_size), key=lambda x: x.fitness)

    def _crossover(self, parent1: Crossword, parent2: Crossword, crossover_rate: float = 0.5) -> Crossword:
        child = copy(parent1)

        for i in range(len(parent1.words)):
            if random.random() < crossover_rate:
                child.words[i].x = parent2.words[i].x
                child.words[i].y = parent2.words[i].y
                child.words[i].direction = parent2.words[i].direction

        return child

    def _mutate_population(self, initial_population: List[Crossword], mutation_rate: float = 0.01):
        return [self._mutate(x, mutation_rate) for x in [copy(crossword) for crossword in initial_population]]

    def _mutate(self, initial_individual: Crossword, mutation_rate: float = 0.01) -> Crossword:
        individual = copy(initial_individual)

        for word in individual.words:
            if random.random() < mutation_rate:
                mutation_probability = random.random()

                if mutation_probability < 0.33:
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))

                elif mutation_probability < 0.66:
                    word.y = random.randint(0, individual.generate_safe_y_from_word(word))

                else:
                    word.direction = random.choice(list(Direction))
                    if not individual.word_within_bounds(word):
                        word.x = random.randint(0, individual.generate_safe_x_from_word(word))
                        word.y = random.randint(0, individual.generate_safe_y_from_word(word))

        return individual

    def run(self, max_generation=100000, current_generation: int = 0, current_try: int = 0, max_tries: int = 100):
        idle_generations = 0
        max_fitness = MAX_INT
        self.population = self.generate_random_population(self.population_size)

        for generation in range(current_generation, max_generation):
            self.calculate_fitnesses()

            self.population = sorted(self.population, key=lambda x: x.fitness)

            current_best_fitness = self.population[0].fitness

            if current_best_fitness == 0:
                self.population[0].calculate_fitness()
                print(f"Best fitness: {self.population[0].fitness}")
                self.population[0].print()
                return

            print(f"Best fitness: {current_best_fitness}")

            if current_best_fitness == max_fitness:
                idle_generations += 1
            elif current_best_fitness < max_fitness:
                max_fitness = current_best_fitness
                idle_generations = 0

            self.population[0].print()
            print(f"Generation: {generation} and {current_try}'th try")

            if idle_generations >= 5:
                self.population = self._mutate_population(self.population, 0.1)

            if idle_generations >= 50:
                self.run(
                    max_generation=max_generation,
                    current_generation=generation,
                    current_try=current_try + 1,
                    max_tries=max_tries
                )
                break

            self.population = self._selection(self.population)

        self.calculate_fitnesses()


def read_input(inputs: str = "./inputs") -> List[List[str]]:
    tests = []
    for test in filter(lambda x: x.startswith("input") and x.endswith(".txt"), os.listdir(inputs)):
        with open(f"{inputs}/{test}") as file:
            tests.append([x.rstrip("\n") for x in file.readlines()])

    return tests

def main():
    for test in read_input():
        crossword = EvolutionaryAlgorithm(test)
        crossword.run()


if __name__ == "__main__":
    main()
