#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sviatoslav Sviatkin
"""

from __future__ import annotations

import enum
import itertools
import random
import sys
from typing import List

MAX_INT = sys.maxsize


class Direction(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Awards(enum.Enum):
    INTERSECT_WITH_EQUAL_LETTER = 30
    INTERSECT_WITH_DIFFERENT_LETTERS = -10

    NEAR = -8
    FAR = 0


class Letter:
    x: int
    y: int
    value: str

    def __init__(self, x: int, y: int, value: str):
        self.x, self.y = x, y
        self.value = value

    def __eq__(self, other: Letter) -> bool:
        return self.x == other.x and self.y == other.y and self.value == other.value


class Word:
    letters: List[Letter]
    x: int
    y: int
    length: int
    direction: Direction

    def __init__(self, letters: List[Letter], direction: Direction):
        self.letters = letters

        self.x = self.letters[0].x
        self.y = self.letters[0].y

        self.length = len(letters)

        self.direction = direction

    def __repr__(self) -> str:
        return "".join([x.value for x in self.letters])

    def __eq__(self, other: Word) -> bool:
        return self.length == other.length and all([self.letters[i] == other.letters[i] for i in range(self.length)])

    def __intersect(self, other: Word) -> Awards:
        for self_letter in self.letters:
            for other_letter in other.letters:
                if self_letter == other_letter:
                    return Awards.INTERSECT_WITH_EQUAL_LETTER
                else:
                    if (self_letter.x == other_letter.x) and (self_letter.y == other_letter.y):
                        return Awards.INTERSECT_WITH_DIFFERENT_LETTERS

        return Awards.FAR

    def __min_distance(self, other: Word) -> int:
        min_distance = MAX_INT
        for letter1 in self.letters:
            for letter2 in other.letters:
                distance = abs(letter1.x - letter2.x) + abs(letter1.y - letter2.y)
                min_distance = min(min_distance, distance)
        return min_distance

    def status(self, other: Word) -> int:
        min_distance = self.__min_distance(other)

        if min_distance == 1:
            return Awards.NEAR.value
        elif min_distance < 1:
            return self.__intersect(other).value
        else:
            return Awards.FAR.value

    @staticmethod
    def create_from_string(string: str, x: int, y: int, direction: Direction) -> Word:
        letters = []
        for index, letter in enumerate(string):
            shift_x = index if direction == Direction.HORIZONTAL else 0
            shift_y = index if direction == Direction.VERTICAL else 0

            letters.append(Letter(x + shift_x, y + shift_y, letter))

        return Word(letters, direction)

    def move(self, new_x: int, new_y: int, new_direction: Direction) -> None:
        for index, letter in enumerate(self.letters):
            shift_x = index if new_direction == Direction.HORIZONTAL else 0
            shift_y = index if new_direction == Direction.VERTICAL else 0

            letter.x = new_x + shift_x
            letter.y = new_y + shift_y


class Crossword:
    n: int
    m: int

    words: List[Word]

    def __init__(self, n: int = 20, m: int = 20, words=None):
        if words is None:
            words = []

        self.words = words

        self.n = n
        self.m = m

    def visualize(self) -> None:
        grid = [[' ' for _ in range(self.m)] for _ in range(self.n)]

        for word in self.words:
            for letter in word.letters:
                grid[letter.x][letter.y] = letter.value

        print("- " * (self.n + 1))
        for row in grid:
            print(f"|{' '.join(row)}|")
        print("- " * (self.n + 1))

        for word1, word2 in itertools.combinations(self.words, 2):
            print(f"{word1} and {word2}: {Awards(word1.status(word2))}")

    def validate_word_location(self, word: Word) -> bool:
        first_letter = word.letters[0]
        last_letter = word.letters[-1]

        for coord in (first_letter.x, first_letter.y, last_letter.x, last_letter.y):
            if (not (0 <= coord < self.n)) or (not (0 <= coord < self.m)):
                return False

        return True

    def generate_randon_locations(self, strings: List[str]) -> List[Word]:
        words = []

        for string in strings:
            direction = random.choice(list(Direction))

            constraint_x = self.n - 1
            constraint_y = self.m - 1

            if direction == Direction.HORIZONTAL:
                constraint_x = constraint_x - len(string)
            if direction == Direction.VERTICAL:
                constraint_y = constraint_y - len(string)

            x = random.randint(0, constraint_x)
            y = random.randint(0, constraint_y)

            word = Word.create_from_string(string, x, y, direction)
            words.append(word)

        return words


class EvolutionaryAlgorithm:
    populations: List[Crossword]

    strings: List[str]

    n: int
    m: int

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self.strings = strings
        self.n = n
        self.m = m

        self.populations = []

        initial = Crossword(n=n, m=m)
        initial.words = initial.generate_randon_locations(strings)

        self.populations.append(initial)

    @staticmethod
    def fitness(crossword: Crossword) -> int:
        population = crossword.words

        # Award points for each word placed
        score = len(population) * 10

        # Penalize words not fully inside grid
        for individual in population:
            if not crossword.validate_word_location(individual):
                score -= 15

        # Award points for each overlapping letter match
        for word1, word2 in itertools.combinations(population, 2):
            if word1 == word2:
                continue

            score += word1.status(word2)

        return score

    @staticmethod
    def selection(self, population: List[Crossword]):

        # Tournament selection
        next_generation = []
        tournament_size = 3

        # Calculate fitness first
        fitnesses = []
        for individual in population:
            fitnesses.append(self.fitness(individual))

        for _ in range(len(population)):
            # Select individuals based on tournament
            tournament = random.sample(list(zip(population, fitnesses)), k=tournament_size)

            # Get the fittest individual
            best = max(tournament, key=lambda x: x[1])[0]

            # Add the winner to next generation
            next_generation.append(best)

        return next_generation


def main() -> None:
    array_of_strings = ["zoo", "goal", "ape"]

    evolution = EvolutionaryAlgorithm(array_of_strings, n=5, m=5)
    evolution.populations[0].visualize()

    print(f"Value of fitness function : {evolution.fitness(evolution.populations[0])}")


if __name__ == "__main__":
    main()
