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


def get_amount_of_combinations(n: int, k: int) -> int:
    return int((math.factorial(n)) / (math.factorial(k) * math.factorial(n - k)))


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
    __x: int
    __y: int
    __direction: Direction
    __value: str
    __length: int

    __genetic_string: str

    def __init__(self, x: int, y: int, direction: Direction, value: str) -> None:
        self.__x = x
        self.__y = y
        self.__direction = direction
        self.__value = value
        self.__length = len(value)

        self.__genetic_string = self.to_genetic_string(x, y, direction)

    def __repr__(self) -> str:
        return f"Word<value={self.value};x={self.x};y={self.y};direction={self.direction}>"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Word) -> bool:
        return (self.x == other.x) and (self.y == other.y) and (self.value == other.value)

    @property
    def x(self) -> int:
        return self.__x

    @x.setter
    def x(self, x: int) -> None:
        self.__x = x
        self.__update_genetic_string()

    @property
    def y(self) -> int:
        return self.__y

    @y.setter
    def y(self, y: int) -> None:
        self.__y = y
        self.__update_genetic_string()

    @property
    def direction(self) -> Direction:
        return self.__direction

    @direction.setter
    def direction(self, direction: Direction) -> None:
        self.__direction = direction
        self.__update_genetic_string()

    @property
    def value(self) -> str:
        return self.__value

    @property
    def length(self) -> int:
        return self.__length

    @property
    def genetic_string(self) -> str:
        return self.__genetic_string

    @genetic_string.setter
    def genetic_string(self, genetic_string: str) -> None:
        self.__genetic_string = genetic_string
        self.__update_values_from_genetic_string()

    def __update_genetic_string(self) -> None:
        self.__genetic_string = self.to_genetic_string(self.x, self.y, self.direction)

    def __update_values_from_genetic_string(self) -> None:
        x, y, direction = self.parse_genetic_string(self.genetic_string)
        self.__x = x
        self.__y = y
        self.__direction = direction

    @staticmethod
    def parse_genetic_string(genetic_string: str) -> tuple[int, int, Direction]:
        x, y, direction = genetic_string.split(",")

        return int(x), int(y), Direction(direction)

    @staticmethod
    def to_genetic_string(x: int, y: int, direction: Direction) -> str:
        return f"{x},{y},{direction.value}"


class Crossword:
    __n: int
    __m: int
    __words: List[Word]
    __strings: List[str]

    __genetic_string: str

    def __init__(self, strings: List[str], n: int = 20, m: int = 20) -> None:
        self.__strings = strings
        self.__n = n
        self.__m = m

        self.words = self.create_random_crossword()



    def create_random_crossword(self) -> List[Word]:
        words = []

        for string in self.strings:
            random_direction = random.choice(list(Direction))
            random_x = random.randint(0, self.n - 1 - (len(string) if random_direction == Direction.HORIZONTAL else 0))
            random_y = random.randint(0, self.m - 1 - (len(string) if random_direction == Direction.VERTICAL else 0))

            words.append(Word(random_x, random_y, random_direction, string))

        return words

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

    def check_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.m

    def get_genetic_string(self) -> str:
        return ";".join([x.genetic_string for x in self.words])

    def update_from_genetic_string(self, genetic_string: str) -> None:
        genetic_words = genetic_string.split(";")
        if len(genetic_words) != len(self.words):
            raise CrosswordUpdateException(
                "Can't update crossword from this genetic string - number of words is different")

        for index, word in enumerate(genetic_words):
            self.words[index].parse_genetic_string(word)

    @property
    def n(self) -> int:
        return self.__n

    @property
    def m(self) -> int:
        return self.__m

    @property
    def strings(self) -> List[str]:
        return self.__strings

    @property
    def words(self) -> List[Word]:
        return self.__words

    @words.setter
    def words(self, words: List[Word]) -> None:
        self.__words = words
        self.__genetic_string = self.get_genetic_string()


class EvolutionaryAlgorithm:
    __strings: List[str]
    __population: List[Crossword]

    __n: int
    __m: int

    def __init__(self, strings: List[str], n: int = 20, m: int = 20) -> None:
        self.__strings = strings
        self.__n = n
        self.__m = m

        self.__population = []

    def generate_random_population(self, population_size: int = 100) -> List[Crossword]:
        return [Crossword(self.strings, self.n, self.m) for _ in range(population_size)]

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

    def __validate_word_location(self, word: Word) -> bool:
        for i in range(word.length):
            x = (word.x + i) if word.direction == Direction.HORIZONTAL else word.x
            y = (word.y + i) if word.direction == Direction.VERTICAL else word.y

            if (not (0 <= x < self.n)) or (not (0 <= y < self.m)):
                return False

        return True

    def crossover(self, parent1: Crossword, parent2: Crossword, crossover_rate: float = 0.5) -> Crossword:
        child = copy.deepcopy(parent1)

        for i in range(len(child.words)):
            if random.random() < crossover_rate:
                child.words[i].x = parent2.words[i].x
                child.words[i].y = parent2.words[i].y
                child.words[i].direction = parent2.words[i].direction

        return child

    def mutation(self, initial_population: List[Crossword], mutation_rate: float = 0.5) -> List[Crossword]:
        population = copy.deepcopy(initial_population)

        for crossword in population:
            for word in crossword.words:

                constraint_x = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
                constraint_y = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)

                # randomly change X gene
                if random.random() < mutation_rate:
                    word.x = random.randint(0, constraint_x)

                # randomly change Y gene
                if random.random() < mutation_rate:
                    word.y = random.randint(0, constraint_y)

                # randomly change Direction gene
                if random.random() < mutation_rate:
                    word.direction = random.choice(list(Direction))

                    # check if word fits field and relocate it if not
                    constraint_x = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
                    if not (0 <= word.x < constraint_x):
                        word.x = random.randint(0, constraint_x)

                    constraint_y = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)
                    if not (0 <= word.y < constraint_y):
                        word.y = random.randint(0, constraint_y)

        return population

    def tournament_selection(self, initial_population: List[Crossword], tournament_size: int = 3) -> Crossword:
        tournament = random.sample(initial_population, k=tournament_size)

        return max(tournament, key=lambda x: self.fitness(x))

    def selection(self, best_individuals_percentage=0.2) -> List[Crossword]:

        best_individuals = self.population[:int(len(self.population) * best_individuals_percentage)]

        # select the rest of the individuals randomly
        rest_individuals_len = len(self.population) - int(len(self.population) * best_individuals_percentage)
        rest_individuals = random.sample(self.population[:], rest_individuals_len)

        # perform crossover of all the individuals by pairing them randomly
        new_individuals = []
        for i in range(len(rest_individuals)):
            new_individuals.append(
                self.crossover(self.tournament_selection(self.population), self.tournament_selection(self.population)))

        new_individuals = self.mutation(new_individuals, 0.03)
        new_population = best_individuals + new_individuals
        return new_population

    def fitness(self, individual: Crossword, visualize: bool = False) -> int:
        intersections = 0
        incorrect_intersection = 0
        near = 0
        wrong_location = 0
        have_no_intersection = 0

        for word1 in individual.words:
            has_intersection = False
            for word2 in individual.words:
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

            if not self.__validate_word_location(word1):
                wrong_location += 1

        # result = - ((4 ** have_no_intersection) + (wrong_location ** 2) + (near * 10) + (1 ** incorrect_intersection))
        if visualize:
            # print(f"- "
            #       f"((4 ** {have_no_intersection}) + "
            #       f"({wrong_location} ** 5) + "
            #       f"({near} * 10) + "
            #       f"(1 ** {incorrect_intersection})) = "
            #       f"{result}")

            print(f"Fitness : {- (have_no_intersection + wrong_location + near + incorrect_intersection)}")
            print(f"Population is {len(self.population)}")
        return - (have_no_intersection + wrong_location + near + incorrect_intersection)

    def run(self, population_size: int = 100, max_generations: int = 100000) -> None:
        best_fitness = -MAX_INT
        stagnant_generations = 0
        self.population = self.generate_random_population(population_size)

        for generation in range(max_generations):
            print(f"Generation {generation}")

            # sort the population by fitness
            self.population.sort(key=lambda x: self.fitness(x))

            current_best_fitness = self.fitness(self.population[0])
            print(f"Best fitness: {current_best_fitness}")
            self.population[0].print()

            if current_best_fitness == 0:
                break

            if current_best_fitness == best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = current_best_fitness
                stagnant_generations = 0

            # shake the population
            if stagnant_generations >= 50:
                self.mutation(self.population, 0.3)
            if stagnant_generations >= 100:
                self.mutation(self.population, 0.9)

            self.selection()

    @property
    def n(self) -> int:
        return self.__n

    @property
    def m(self) -> int:
        return self.__m

    @property
    def strings(self) -> List[str]:
        return self.__strings

    @property
    def population(self) -> List[Crossword]:
        return self.__population

    @population.setter
    def population(self, population: List[Crossword]) -> None:
        self.__population = population


def main() -> None:
    array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]
    evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)

    # array_of_strings = ["zoo", "goal", "ape"]
    # evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)

    evolution.run()


if __name__ == "__main__":
    main()
