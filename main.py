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

    def __init__(self, x: int, y: int, direction: Direction, value: str):
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
    def x(self, x: int):
        self.__x = x
        self.__update_genetic_string()

    @property
    def y(self) -> int:
        return self.__y

    @y.setter
    def y(self, y: int):
        self.__y = y
        self.__update_genetic_string()

    @property
    def direction(self) -> Direction:
        return self.__direction

    @direction.setter
    def direction(self, direction: Direction):
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
    def genetic_string(self, genetic_string: str):
        self.__genetic_string = genetic_string
        self.__update_values_from_genetic_string()

    def __update_genetic_string(self) -> None:
        self.__genetic_string = self.to_genetic_string(self.x, self.y, self.direction)

    def __update_values_from_genetic_string(self):
        x, y, direction = self.parse_genetic_string(self.genetic_string)
        self.__x = x
        self.__y = y
        self.__direction = direction

    @staticmethod
    def parse_genetic_string(genetic_string: str) -> tuple[int, int, Direction]:
        x, y, direction = genetic_string.split(",")

        return int(x), int(y), Direction(direction)

    @staticmethod
    def to_genetic_string(x: int, y: int, direction: Direction):
        return f"{x},{y},{direction.value}"


class Crossword:
    __n: int
    __m: int
    __words: List[Word]
    __strings: List[str]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self.__strings = strings
        self.__n = n
        self.__m = m

        self.__words = self.create_random_crossword()

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
    def words(self):
        return self.__words


def main() -> None:
    # array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]
    array_of_strings = ["zoo", "goal", "ape"]

    crossword = Crossword(array_of_strings, n=5, m=5)
    crossword.print()

    # evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)
    #
    # evolution.run()


if __name__ == "__main__":
    main()
