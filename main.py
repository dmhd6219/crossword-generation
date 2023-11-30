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
        return f"{self.x},{self.y},{self.direction.value}"

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

    def __str__(self) -> str:
        return ";".join([str(x) for x in self.words])

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
        return self._fitness

    def calculate_fitness(self) -> None:
        intersections = 0
        incorrect_intersection = 0
        near = 0
        wrong_location = 0
        have_no_intersection = 0

        for word1 in self.words:
            has_intersection = False
            for word2 in self.words:
                if word1 == word2:
                    continue

                distance = self.__calc_distance(word1, word2)
                if distance == 0:
                    intersections += 1

                    intersection_coords = self.__get_intersection_coords(word1, word2)
                    if len(intersection_coords) > 1:
                        incorrect_intersection += len(intersection_coords)
                    else:
                        if word1.value[intersection_coords[0][0]] != word2.value[intersection_coords[0][1]]:
                            incorrect_intersection += 1

                elif distance == 1:
                    near += 1

            if not has_intersection:
                have_no_intersection += 1

            if not self.check_word_bounds(word1):
                wrong_location += 1

        self._fitness = -(incorrect_intersection + near + wrong_location + have_no_intersection)

    @staticmethod
    def __calc_distance(word1: Word, word2: Word) -> int:
        min_dist = MAX_INT

        for i1 in range(word1.length):
            for i2 in range(word2.length):
                x1 = (word1.x + i1) if word1.direction == Direction.HORIZONTAL else word1.x
                y1 = (word1.y + i1) if word1.direction == Direction.VERTICAL else word1.y
                x2 = (word2.x + i2) if word2.direction == Direction.HORIZONTAL else word2.x
                y2 = (word2.y + i2) if word2.direction == Direction.VERTICAL else word2.y

                dist_x = abs(x1 - x2) if y1 == y2 else MAX_INT
                dist_y = abs(y1 - y2) if x1 == x2 else MAX_INT

                min_dist = min(min_dist, dist_x, dist_y)

        return min_dist

    @staticmethod
    def __get_intersection_coords(word1: Word, word2: Word) -> List[tuple[int, int]]:
        # TODO : check on correctness
        coords = []
        for i1 in range(word1.length):
            for i2 in range(word2.length):
                x1 = (word1.x + i1) if word1.direction == Direction.HORIZONTAL else word1.x
                y1 = (word1.y + i1) if word1.direction == Direction.VERTICAL else word1.y
                x2 = (word2.x + i2) if word2.direction == Direction.HORIZONTAL else word2.x
                y2 = (word2.y + i2) if word2.direction == Direction.VERTICAL else word2.y

                if (x1 == x2) and (y1 == y2):
                    coords.append((i1, i2))

        return coords

    def check_bounds(self, x: int, y: int):
        return (0 <= x < self.n) and (0 <= y < self.m)

    def check_word_bounds(self, word: Word) -> bool:
        x = word.x
        y = word.y

        return self.check_bounds(x, y)

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

    def generate_random_population(self, population_size: int) -> List[Crossword]:
        return [Crossword(self.strings, self.n, self.m) for _ in range(population_size)]

    @staticmethod
    def crossover(parent1: Crossword, parent2: Crossword) -> Crossword:
        number_of_genes_to_exchange = random.randint(1, len(parent1.words) - 1)

        child = copy.copy(parent1)

        for i in range(number_of_genes_to_exchange):
            index_to_exchange = random.randint(0, len(parent1.words) - 1)

            child.words[index_to_exchange] = copy.copy(parent2.words[index_to_exchange])

        # print(f"------\n Crossover: {str(parent1)} + {str(parent2)} = {str(child)}\n -----------")
        return child

    def mutation(self, initial_population: List[Crossword], mutation_rate: float = 0.5) -> List[Crossword]:
        population = []

        for crossword in initial_population:
            new_crossword = copy.copy(crossword)
            new_words = []

            for word in crossword.words:
                new_word = copy.copy(word)

                if random.random() < mutation_rate:
                    if new_word.direction == Direction.VERTICAL:
                        new_word.x = random.randint(0, self.n - new_word.length)
                    else:
                        new_word.x = random.randint(0, self.n - 1)

                if random.random() < mutation_rate:
                    if new_word.direction == Direction.HORIZONTAL:
                        new_word.y = random.randint(0, self.m - new_word.length)
                    else:
                        new_word.y = random.randint(0, self.m - 1)

                if random.random() < mutation_rate:
                    new_word.direction = random.choice(list(Direction))
                    if new_word.direction == Direction.VERTICAL:
                        if new_word.y + new_word.length > self.m:
                            new_word.y = random.randint(0, self.m - new_word.length)
                    else:
                        if new_word.x + new_word.length > self.n:
                            new_word.x = random.randint(0, self.n - new_word.length)
                new_words.append(new_word)

            new_crossword.words = new_words
            population.append(new_crossword)

        # print(f"-------\n Mutation: \n{[str(x) for x in initial_population]} -> \n{[str(x) for x in population]} \n------")
        # print([str(initial_population[i]) == str(population[i]) for i in range(len(initial_population))])

        return population

    @staticmethod
    def _tournament_selection(population: List[Crossword]) -> Crossword:
        return max(random.sample(population, 3), key=lambda x: x.fitness)

    def selection(self, initial_population: List[Crossword], percentage: float = 0.2) -> List[Crossword]:
        population = copy.copy(sorted(initial_population, key=lambda x: x.fitness))
        best_individuals = population[:int(len(population) * percentage)]

        rest_individuals_len = len(population) - int(len(population) * percentage)
        rest_individuals = random.sample(population[:], rest_individuals_len)

        new_individuals = []
        for i in range(len(rest_individuals)):
            new_individuals.append(
                self.crossover(self._tournament_selection(population), self._tournament_selection(population)))

        new_individuals = self.mutation(new_individuals, 0.03)
        new_population = best_individuals + new_individuals

        # print(
        #     f"-------\n Selection: \n{[str(x) for x in initial_population]} -> \n{[str(x) for x in population]} \n------")
        # print([str(initial_population[i]) == str(population[i]) for i in range(len(initial_population))])
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

        self.population = self.generate_random_population(population_size)

        for generation in range(max_generations):
            print(f"Generation {generation}")

            self.update_fitness()
            best_crossword = self.find_best_crossword()
            print(f"Best fitness : {best_crossword.fitness}")
            best_crossword.print()

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

            self.population = self.selection(self.population)

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
