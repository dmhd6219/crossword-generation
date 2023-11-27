#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sviatoslav Sviatkin
"""

from __future__ import annotations

import enum
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
    __strings: List[str]
    __x: int
    __y: int
    __direction: Direction

    __genetic_string: str

    def __init__(self, x: int, y: int, direction: Direction):
        self.__x = x
        self.__y = y
        self.__direction = direction
        self.__genetic_string = self.to_genetic_string(x, y, direction)

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
        return f"{x},{y},{direction}"


def main() -> None:
    array_of_strings = ["wonderful", "goal", "lame", "fullstack", "wario", "organ", "nigger"]

    # evolution = EvolutionaryAlgorithm(array_of_strings, n=20, m=20)
    #
    # evolution.run()


if __name__ == "__main__":
    main()
