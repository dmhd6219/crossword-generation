from __future__ import annotations

import enum
import random
from typing import List


class Direction(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Awards(enum.Enum):
    INTERSECT_WITH_EQUAL_LETTER = 10
    INTERSECT_WITH_DIFFERENT_LETTERS = -10
    DONT_INTERSECT = 0

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

    def intersect(self, other: Word) -> int:
        for self_letter in self.letters:
            for other_letter in other.letters:
                if self_letter == other_letter:
                    return Awards.INTERSECT_WITH_EQUAL_LETTER.value
                else:
                    if (self.x == other.x) and (self.y == other.y):
                        return Awards.INTERSECT_WITH_DIFFERENT_LETTERS.value

        return Awards.DONT_INTERSECT.value

    def near(self, other: Word) -> int:
        # TODO implement detecting near words
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
                print("wrong!! " + str(word))
                return False

        return True

    def generate_randon_locations(self) -> None:
        # TODO: fix errors with placing out of field
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
        for word1 in self.words:
            for word2 in self.words:
                if word1 == word2:
                    continue

                score += word1.intersect(word2)

        return score


def main() -> None:
    array_of_strings = ["zoo", "goal", "ape"]

    crossword = Crossword(array_of_strings, n=5, m=5)
    crossword.visualize()

    print(crossword.fitness())


if __name__ == "__main__":
    main()
