from __future__ import annotations

import enum
import random
from typing import List


class Direction(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Letter:
    x: int
    y: int
    value: str

    def __init__(self, x: int, y: int, value: str):
        self.x, self.y = x, y
        self.value = value


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
    strings: List[str]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20):
        self.strings = strings
        self.words = []

        self.n = n
        self.m = m

        self.generate_randon_locations()

    def visualize(self) -> None:
        grid = [[' ' for _ in range(self.m)] for _ in range(self.n)]

        for word in self.words:
            for letter in word.letters:
                grid[letter.x][letter.y] = letter.value

        print("- " * (self.n + 1))
        for row in grid:
            print(f"|{' '.join(row)}|")
        print("- " * (self.n + 1))

    def validate_word_location(self, word: Word) -> bool:
        first_letter = word.letters[0]
        last_letter = word.letters[-1]

        for coord in (first_letter.x, first_letter.y, last_letter.x, last_letter.y):
            if (not (0 <= coord < self.n)) or (not (0 <= coord < self.m)):
                return False

        return True

    def generate_randon_locations(self) -> None:
        self.words = []

        for string in self.strings:
            direction = random.choice(list(Direction))

            constraint_x = self.n
            constraint_y = self.m

            if direction == Direction.HORIZONTAL:
                constraint_x = constraint_x - len(string)
            if direction == Direction.VERTICAL:
                constraint_y = constraint_y - len(string)

            x = random.randint(0, constraint_x)
            y = random.randint(0, constraint_y)

            word = Word.create_from_string(string, x, y, direction)
            self.words.append(word)

    def fitness(self) -> int:
        # Award points for each word placed
        score = len(self.words) * 10

        # Penalize words not fully inside grid
        for word in self.words:
            if not self.validate_word_location(word):
                score -= 5

        # Award points for each overlapping letter match
        letter_positions = set()
        for word in self.words:
            for letter in word.letters:
                if (letter.x, letter.y) in letter_positions:
                    score += 3
                letter_positions.add((letter.x, letter.y))

        # Check vertical separation
        for i in range(len(self.words)):
            for j in range(i + 1, len(self.words)):
                w1 = self.words[i]
                w2 = self.words[j]
                if w1.direction == w2.direction == Direction.VERTICAL:
                    if abs(w1.x - w2.x) < 2:
                        score -= 15

        # Check horizontal separation
        for i in range(len(self.words)):
            for j in range(i + 1, len(self.words)):
                w1 = self.words[i]
                w2 = self.words[j]
                if w1.direction == w2.direction == Direction.HORIZONTAL:
                    if abs(w1.y - w2.y) < 2:
                        score -= 15

        return score


def main() -> None:
    array_of_strings = ["qwerty", "ask", "sviatoslav"]

    crossword = Crossword(array_of_strings)
    crossword.visualize()


if __name__ == "__main__":
    main()
