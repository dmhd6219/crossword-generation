#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sviatoslav Sviatkin
"""

from __future__ import annotations

import copy
import enum
import itertools
import math
import random
import sys
from typing import List

MAX_INT = sys.maxsize


class CrosswordUpdateException(Exception):
    pass


class Direction(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Awards(enum.Enum):
    INTERSECT_WITH_EQUAL_LETTER = 0
    INTERSECT_WITH_DIFFERENT_LETTERS = -10

    NEAR = -8
    FAR = 0


class Word:
    x: int
    y: int
    direction: Direction
    _value: str
    _length: int

    def __init__(self, x: int, y: int, direction: Direction, value: str):
        self.x = x
        self.y = y
        self.direction = direction
        self._value = value
        self._length = len(value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Word<x=[{self.x}], y=[{self.y}], direction=[{self.direction}], value=[{self.value}]>"

    def __copy__(self) -> Word:
        return Word(self.x, self.y, self.direction, self.value)

    @property
    def value(self) -> str:
        return self._value

    @property
    def length(self) -> int:
        return self._length


class Crossword:
    _n: int
    _m: int
    _strings: List[str]
    words: List[Word]

    _fitness: int

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self._strings = strings
        self._n = n
        self._m = m

        self.words = self._generate_random_positions()
        self._fitness = -MAX_INT

    def __copy__(self) -> Crossword:
        crossword = Crossword(self.strings, self.n, self.m)
        crossword.words = [copy.copy(x) for x in self.words]
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

    @property
    def fitness(self) -> int:
        return self.fitness

    def calculate_fitness(self) -> int:
        # TODO
        for _ in self.words:
            pass
        return -MAX_INT

    def check_bounds(self, x: int, y: int):
        return (0 <= x < self.n) and (0 <= y < self.m)

    def print(self) -> None:
        matrix = [[" . " for _ in range(0, self.m)] for _ in range(0, self.n)]

        for word in self.words:
            for i in range(word.length):
                new_x = word.x + (i if word.direction == Direction.HORIZONTAL else 0)
                new_y = word.y + (i if word.direction == Direction.VERTICAL else 0)
                if self.check_bounds(new_x, new_y):
                    matrix[new_x][new_y] = f" {word.value[i]} "

        print(" " + " - " * self.m + " ")
        for row in matrix:
            print(f'|{"".join(row)}|')
        print(" " + " - " * self.m + " ")

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


class EvolutionaryAlgorithm:
    _strings: List[str]
    population: List[Crossword]

    _n: int
    _m: int

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self._strings = strings
        self._n = n
        self._m = m
        self.population = []

    @property
    def strings(self) -> List[str]:
        return self._strings

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @staticmethod
    def crossover(parent1: Crossword, parent2: Crossword) -> Crossword:
        number_of_genes_to_exchange = random.randint(1, len(parent1.words) - 1)

        child = copy.copy(parent1)

        for i in range(number_of_genes_to_exchange):
            index_to_exchange = random.randint(0, len(parent1.words) - 1)

            child.words[index_to_exchange] = copy.copy(parent2.words[index_to_exchange])
        return child

    def mutation(self, initial_population: List[Crossword], mutation_rate: float = 0.5) -> List[Crossword]:
        population = copy.copy(initial_population)

        for crossword in population:
            for word in crossword.words:

                if random.random() < mutation_rate:
                    if word.direction == Direction.VERTICAL:
                        word.x = random.randint(0, self.n - word.length)
                    else:
                        word.x = random.randint(0, self.n - 1)

                if random.random() < mutation_rate:
                    if word.direction == Direction.HORIZONTAL:
                        word.y = random.randint(0, self.m - word.length)
                    else:
                        word.y = random.randint(0, self.m - 1)

                if random.random() < mutation_rate:
                    word.direction = random.choice(list(Direction))
                    if word.direction == Direction.VERTICAL:
                        if word.y_position + word.length > self.m:
                            word.y_position = random.randint(0, self.m - word.length)
                    else:
                        if word.x_position + word.length > self.n:
                            word.x_position = random.randint(0, self.n - word.length)

        return population

    @staticmethod
    def _tournament_selection(population: List[Crossword]) -> Crossword:
        return max(random.sample(population, 3), key=lambda x: x.fitness)

    def selection(self, percentage: float = 0.2) -> List[Crossword]:
        population = copy.copy(sorted(self.population, key=lambda x: x.fitness))
        best_individuals = population[:int(len(population) * percentage)]

        rest_individuals_len = len(population) - int(len(population) * percentage)
        rest_individuals = random.sample(population[:], rest_individuals_len)

        new_individuals = []
        for i in range(len(rest_individuals)):
            new_individuals.append(
                self.crossover(self._tournament_selection(population), self._tournament_selection(population)))

        new_individuals = self.mutation(new_individuals, 0.03)
        new_population = best_individuals + new_individuals
        return new_population

    def update_fitness(self):
        for crossword in self.population:
            crossword.calculate_fitness()

    def find_best_crossword(self, visualize: bool = False) -> Crossword:
        population = sorted(self.population, key=lambda x: x.fitness)
        current_best_fitness = population[0].fitness

        if visualize:
            print(f"Best fitness: {current_best_fitness}")
            self.population[0].print()

        return population[0]

    def run(self, population_size: int = 100, max_generations: int = 1000, depth: int = 0):
        best_fitness = MAX_INT
        stagnant_generations = 0

        for generation in range(max_generations):
            print(f"Generation {generation}")

            self.update_fitness()
            best_crossword = self.find_best_crossword()

            if best_crossword.fitness == 0:
                break

            if best_crossword.fitness == best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = best_crossword.fitness
                stagnant_generations = 0

            if stagnant_generations > 300:
                self.run(population_size=population_size, max_generations=max_generations, depth=depth + 1)

            if depth >= 5:
                return

            self.population = self.selection()

        self.update_fitness()
        self.find_best_crossword(visualize=True)


def main() -> None:
    array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]
    evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)

    # array_of_strings = ["zoo", "goal", "ape"]
    # evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)

    evolution.run()


if __name__ == "__main__":
    main()
