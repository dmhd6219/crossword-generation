#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sviatoslav Sviatkin
"""

from __future__ import annotations

import enum
import random
import sys
from typing import List

MAX_INT = sys.maxsize


class BadCombinationException(Exception):
    pass


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

    def run(self, population_size: int = 100, max_generations: int = 100000) -> None:
        self.__population = self.generate_random_population(population_size)

        for generation in range(max_generations):
            print(f"Generation {generation}")

            self.__population.sort(key=lambda x: self.fitness(x))

            best_individual = self.__population[0]
            print(f"Best fitness: {self.fitness(best_individual)}")

            next_population = self.__population[:2]  # Elitism

            for _ in range((population_size - len(next_population)) // 2):
                parent1 = self.tournament_selection(self.__population)
                parent2 = self.tournament_selection(self.__population)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                next_population += [child1, child2]

            self.__population = next_population

        best = self.__population[0]
        print("Best solution found:")
        best.print()

    def tournament_selection(self, population: List[Crossword]):
        # Perform tournament
        competitors = random.sample(population, k=2)
        best = max(competitors, key=self.fitness)
        return best

    @property
    def population(self) -> List[Crossword]:
        return self.__population

    @population.setter
    def population(self, population: List[Crossword]) -> None:
        self.__population = population

    def check_intersection(self, word1: Word, word2: Word) -> bool:
        # Check if words are perpendicular
        if word1.direction == word2.direction:
            return False

        # Calculate word boundaries
        x1_start = word1.x
        x1_end = word1.x + (word1.length if word1.direction == Direction.HORIZONTAL else 0)
        y1_start = word1.y
        y1_end = word1.y + (word1.length if word1.direction == Direction.VERTICAL else 0)

        x2_start = word2.x
        x2_end = word2.x + (word2.length if word2.direction == Direction.HORIZONTAL else 0)
        y2_start = word2.y
        y2_end = word2.y + (word2.length if word2.direction == Direction.VERTICAL else 0)

        # Check if boundaries intersect
        if (x1_start <= x2_start <= x1_end or
                x2_start <= x1_start <= x2_end):
            return True
        if (y1_start <= y2_start <= y1_end or
                y2_start <= y1_start <= y2_end):
            return True

        return False

    def calc_distance(self, word1: Word, word2: Word) -> int:
        min_dist = MAX_INT

        for i1 in range(word1.length):
            for i2 in range(word2.length):
                x1 = word1.x + i1 if word1.direction == Direction.HORIZONTAL else word1.x
                y1 = word1.y + i1 if word1.direction == Direction.VERTICAL else word1.y
                x2 = word2.x + i2 if word2.direction == Direction.HORIZONTAL else word2.x
                y2 = word2.y + i2 if word2.direction == Direction.VERTICAL else word2.y

                dist_x = abs(x1 - x2)
                dist_y = abs(y1 - y2)
                dist = dist_x + dist_y

                min_dist = min(min_dist, dist)

        return min_dist

    def fitness(self, individual: Crossword) -> int:
        fitness = 0

        # Reward intersecting words
        for word1 in individual.words:
            for word2 in individual.words:
                if word1 != word2:
                    if self.check_intersection(word1, word2):
                        if word1.value[0] == word2.value[0]:
                            fitness += Awards.INTERSECT_WITH_EQUAL_LETTER.value
                        else:
                            fitness += Awards.INTERSECT_WITH_DIFFERENT_LETTERS.value

        # Penalize words being too close
        for i in range(len(individual.words)):
            for j in range(i + 1, len(individual.words)):
                word1 = individual.words[i]
                word2 = individual.words[j]

                dist = self.calc_distance(word1, word2)
                if dist <= 2:
                    fitness += Awards.NEAR.value
                else:
                    fitness += Awards.FAR.value

        return fitness

    def crossover(self, parent1: Crossword, parent2: Crossword) -> tuple[Crossword, Crossword]:
        child1 = Crossword(self.strings, self.n, self.m)
        child2 = Crossword(self.strings, self.n, self.m)

        if len(parent1.words) != len(parent2.words):
            raise CrosswordUpdateException("Can't make crossover from two Crosswords with different number of words")

        midpoint = random.randint(1, len(parent1.words) - 2)

        child1.words = parent1.words[:midpoint]
        child1.words += parent2.words[midpoint:]

        child2.words = parent1.words[midpoint:]
        child2.words += parent2.words[:midpoint]

        return child1, child2

    def mutation(self, individual: Crossword) -> Crossword:
        word = random.choice(individual.words)

        if random.random() < 0.33:
            word.x = random.randint(0, self.n - 1)
        elif random.random() < 0.66:
            word.y += random.randint(0, self.m - 1)
        else:
            word.direction = random.choice(list(Direction))

        return individual

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


def main() -> None:
    # array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]
    array_of_strings = ["zoo", "goal", "ape"]

    evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)
    evolution.run()


if __name__ == "__main__":
    main()
