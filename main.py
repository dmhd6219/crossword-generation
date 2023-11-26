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
    INTERSECT_THROUGH_OTHER_WORDS = 1
    INTERSECT_WITH_EQUAL_LETTER = 0
    INTERSECT_WITH_DIFFERENT_LETTERS = -10

    NEAR = -8
    FAR = -5


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

    def __hash__(self):
        return hash(tuple((letter.x, letter.y, letter.value) for letter in self.letters))

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

        print(f"Words : {self.words}")
        for word1, word2 in itertools.combinations(self.words, 2):
            print(f"{word1} and {word2}: {Awards(word1.status(word2))}")

    def validate_word_location(self, word: Word) -> bool:

        min_x = min(l.x for l in word.letters)
        max_x = max(l.x for l in word.letters)
        if min_x < 0 or max_x >= self.n:
            return False

        min_y = min(l.y for l in word.letters)
        max_y = max(l.y for l in word.letters)
        if min_y < 0 or max_y >= self.m:
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

    def are_words_connected(self, word1: Word, word2: Word) -> bool:
        visited = set()

        def dfs(current_word: Word) -> bool:
            if current_word == word2:
                return True

            visited.add(current_word)

            for other_word in self.words:
                if other_word not in visited:
                    if any(letter1 == letter2 for letter1 in current_word.letters for letter2 in other_word.letters):
                        if dfs(other_word):
                            return True

            return False

        return dfs(word1)


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

        for _ in range(100):
            initial = Crossword(n=n, m=m)
            initial.words = initial.generate_randon_locations(strings)

            self.populations.append(initial)

    @staticmethod
    def fitness(crossword: Crossword) -> int:
        population = crossword.words

        # Award points for each word placed
        # score = len(population) * 10
        score = 0

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

    def selection(self, population: List[Crossword]):

        # Tournament selection
        next_generation = []
        for _ in range(len(population)):

            # Get sample and calculate fitnesses
            tournament = random.sample(population, min(3, len(population)))

            fitnesses = []
            for individual in tournament:
                fitnesses.append(self.fitness(individual))

            # Identify the most fit individual
            best = max(zip(tournament, fitnesses), key=lambda x: x[1])[0]

            next_generation.append(best)

        return next_generation

    def crossover(self, parent1: Crossword, parent2: Crossword) -> tuple[Crossword, Crossword]:

        # Create blank child crosswords
        child1 = Crossword(n=self.n, m=self.m)
        child2 = Crossword(n=self.n, m=self.m)

        # Select random cut point
        cut = random.randint(1, len(parent1.words) - 1)

        # Take first words from parent1
        for word in parent1.words[:cut]:
            letters = [Letter(l.x, l.y, l.value) for l in word.letters]
            child1.words.append(Word(letters, word.direction))

        # Take last words from parent2
        for word in parent2.words[cut:]:
            letters = [Letter(l.x, l.y, l.value) for l in word.letters]
            child1.words.append(Word(letters, word.direction))

        # Vice versa
        for word in parent1.words[cut:]:
            letters = [Letter(l.x, l.y, l.value) for l in word.letters]
            child2.words.append(Word(letters, word.direction))

        for word in parent2.words[:cut]:
            letters = [Letter(l.x, l.y, l.value) for l in word.letters]
            child2.words.append(Word(letters, word.direction))

        return child1, child2

    @staticmethod
    def mutate(crossword: Crossword) -> Crossword:
        # Choose a random word
        word = random.choice(crossword.words)

        # Choose a mutation type randomly
        mutation_type = random.choice(["shift", "swap", "flip"])

        if mutation_type == "shift":
            # Small shift in random direction
            shift = random.randint(-2, 2)
            word.x += shift

        elif mutation_type == "swap":
            # Swap x and y coordinates
            word.move(word.y, word.x, word.direction)

        else:
            # Flip the direction
            if word.direction == Direction.HORIZONTAL:
                word.direction = Direction.VERTICAL
            else:
                word.direction = Direction.HORIZONTAL

        return crossword

    def run(self):

        generations = 0
        best_fitness = -1000

        while generations < 10000 and best_fitness < 0:

            next_gen = []

            # Selection
            parents = self.selection(self.populations)

            # Crossover
            for i in range(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])

                # Mutation
                if random.random() < 0.1:
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                next_gen.extend([child1, child2])

                # Calculate fitness
                for puzzle in next_gen:
                    fit = self.fitness(puzzle)
                    if fit > best_fitness:
                        best_fitness = fit

                # Next generation
                self.populations = next_gen

            generations += 1

        random.choice(self.populations).visualize()

        print(generations, best_fitness)


def main() -> None:
    array_of_strings = ["zoo", "goal", "ape"]

    evolution = EvolutionaryAlgorithm(array_of_strings, n=5, m=5)
    random_choice = random.choice(evolution.populations)

    random_choice.visualize()

    print(f"Value of fitness function : {evolution.fitness(random_choice)}")

    for word1, word2 in itertools.combinations(random_choice.words, 2):
        print(f"{word1} and {word2} are connected? {random_choice.are_words_connected(word1, word2)}")


    evolution.run()


if __name__ == "__main__":
    main()
