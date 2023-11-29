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

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self._strings = strings
        self._n = n
        self._m = m
        self.words = self._generate_random_positions()

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def strings(self) -> List[str]:
        return self._strings

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

    _population_size: int

    _n: int
    _m: int

    def __init__(self, strings: List[str], population_size: int = 100, n: int = 20, m: int = 20):
        self._strings = strings
        self._population_size = population_size
        self._n = n
        self._m = m
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

    def crossover(self):
        pass

    def mutation(self):
        pass

    def selection(self):
        pass

    def fitness(self):
        pass

    def run(self):
        pass


def main() -> None:
    array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]
    evolution = EvolutionaryAlgorithm(array_of_strings, population_size=100, n=20, m=20)

    # array_of_strings = ["zoo", "goal", "ape"]
    # evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)

    evolution.run()


if __name__ == "__main__":
    main()
