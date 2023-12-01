from __future__ import annotations

import sys
from enum import Enum
from copy import deepcopy, copy
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

    def fill_edge(self, u: int, v: int) -> None:
        self._matrix[u][v] = GraphCell.FILLED
        self._matrix[v][u] = GraphCell.FILLED

    def is_connected(self) -> bool:
        return self.get_amount_of_disconnected() == 0

    def get_amount_of_disconnected(self) -> int:
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
        self._strings = strings
        self._n = n
        self._m = m

        self._strings = strings

        self.words = self._generate_random_positions()
        self.update_fitness()

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
        if word1.direction == word2.direction == Direction.VERTICAL:
            if not (abs(word1.x - word2.x) <= 1):
                return False

            if word2.y < word1.y:
                word1, word2 = word2, word1

            return word1.y <= word2.y <= word1.y + word1.length

        elif word1.direction == word2.direction == Direction.HORIZONTAL:
            if not abs(word1.y - word2.y) <= 1:
                return False

            if word2.x < word1.x:
                word1, word2 = word2, word1

            return word1.x <= word2.x <= word1.x + word1.length

        return False

    def check_intersection(self, word1: Word, word2: Word) -> bool:
        if word1.direction == Direction.VERTICAL:
            return (word2.x <= word1.x < word2.x + len(word2.value) and
                    word1.y <= word2.y < word1.y + len(word1.value))
        else:
            return (word1.x <= word2.x < word1.x + len(word1.value) and
                    word2.y <= word1.y < word2.y + len(word2.value))

    def check_letter_match(self, word1: Word, word2:Word) -> bool:
        if word1.direction == Direction.VERTICAL:
            return word1.value[word2.y - word1.y] == word2.value[word1.x - word2.x]
        else:
            return word1.value[word2.x - word1.x] == word2.value[word1.y - word2.y]

    def check_collisions(self, word1: Word, word2: Word) -> int:
        collisions = 0

        if word1.direction != word2.direction:
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

    def update_fitness(self):
        g = Graph(len(self.words))
        fitness = 0

        for i in range(len(self.words)):
            word1 = self.words[i]

            fitness += 1000 * (not self.word_within_bounds(word1))

            for j in range(i + 1, len(self.words)):
                word2 = self.words[j]

                if word1.direction == word2.direction:
                    fitness += 5 * self.detect_overlapping(word1, word2)
                else:
                    if self.check_intersection(word1, word2):
                        g.fill_edge(i, j)
                        g.fill_edge(j, i)
                        fitness += 3 * (not self.check_letter_match(word1, word2))

                    fitness += self.check_collisions(word1, word2)

        fitness += g.get_amount_of_disconnected()

        self.fitness = fitness


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

        self.population = self.generate_random_population(population_size)

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
            crossword.update_fitness()

    def generate_random_population(self, population_size=300):
        return [Crossword(self.strings, self.m, self.n) for _ in range(population_size)]

    def _selection(self, initial_population: List[Crossword], best_individuals_percentage=0.2):
        population = [copy(crossword) for crossword in initial_population]

        best_individuals = population[:int(len(population) * best_individuals_percentage)]

        rest_individuals_len = len(population) - int(len(population) * best_individuals_percentage)
        rest_individuals = random.sample(population[:], rest_individuals_len)

        new_individuals = [
            self._crossover(
                self._tournament_selection(rest_individuals),
                self._tournament_selection(rest_individuals)
            )
            for _ in range(len(rest_individuals))
        ]

        # perform mutation on the new individuals
        new_individuals = self._mutation(new_individuals)
        new_population = best_individuals + new_individuals
        return new_population

    def _tournament_selection(self, population: List[Crossword], tournament_size=3):
        return min(random.sample(population, k=tournament_size), key=lambda x: x.fitness)

    def _crossover(self, parent1: Crossword, parent2: Crossword, crossover_rate: float = 0.5) -> Crossword:
        child = deepcopy(parent1)

        for i in range(len(parent1.words)):
            if random.random() < crossover_rate:
                child.words[i].x = parent2.words[i].x
                child.words[i].y = parent2.words[i].y
                child.words[i].direction = parent2.words[i].direction

        return child

    def _mutation(self, initial_population: List[Crossword], mutation_rate: float = 0.01):
        population = [copy(crossword) for crossword in initial_population]

        for crossword in population:
            for word in crossword.words:
                if random.random() < mutation_rate:
                    mutation_probability = random.random()

                    if mutation_probability < 0.3:
                        constraint_x = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
                        word.x = random.randint(0, constraint_x)

                    elif mutation_probability < 0.6:
                        constraint_y = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)
                        word.y = random.randint(0, constraint_y)

                    else:
                        word.direction = random.choice(list(Direction))
                        if not crossword.word_within_bounds(word):
                            constraint_x = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
                            constraint_y = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)

                            word.x = random.randint(0, constraint_x)
                            word.y = random.randint(0, constraint_y)

        return population

    def run(self, max_generation=20000):

        for generation in range(max_generation):
            self.calculate_fitnesses()

            self.population.sort(key=lambda x: x.fitness)

            current_best_fitness = self.population[0].fitness
            print(f"Best fitness: {current_best_fitness}")

            self.population[0].print()
            print(f"Generation: {generation}")

            if current_best_fitness == 0:
                break

            self.population = self._selection(self.population)

        self.calculate_fitnesses()

        self.population.sort(key=lambda x: x.fitness)

        self.population[0].update_fitness()
        print(f"Best fitness: {self.population[0].fitness}")
        self.population[0].print()


def main():
    crossword = EvolutionaryAlgorithm(["wonderful", "fullstack", "warioorgan"])
    crossword.run()


if __name__ == "__main__":
    main()
