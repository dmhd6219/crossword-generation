from __future__ import annotations

import logging
import os
import sys
import time
from enum import Enum
from copy import copy
import random
from typing import List, Set, Tuple

logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w")

MAX_INT = sys.maxsize


class Direction(Enum):
    """
    Enum representing direction of words (horizontal or vertical).

    Attributes:
        HORIZONTAL (int): Horizontal direction
        VERTICAL (int): Vertical direction
    """
    HORIZONTAL = 1
    VERTICAL = 0


class GraphCell(Enum):
    """
    Enum representing state of graph cell (empty or filled).

        Attributes:
            EMPTY (int): Empty cell
            FILLED (int): Filled cell
        """
    EMPTY = 0
    FILLED = 1


class Graph:
    """
    Class representing graph of connections between words.

    Attributes:
        _matrix (List[List[GraphCell]]): Adjacency matrix
        _n (int): Size of graph
    """

    _matrix: List[List[GraphCell]]
    _n: int

    def __init__(self, n: int):
        """
        Constructs graph with n x n matrix of empty cells.

        Args:
            n (int): Size of graph
        """
        self._n = n
        self._matrix = [[GraphCell.EMPTY for _ in range(n)] for _ in range(n)]

    def fill_edge(self, u: int, v: int) -> None:
        """
        Fills edge between nodes u and v in graph.

        Args:
            u (int): Node u
            v (int): Node v
        """
        self._matrix[u][v] = GraphCell.FILLED
        self._matrix[v][u] = GraphCell.FILLED

    def get_amount_of_disconnected(self) -> int:
        """
        Gets number of disconnected subgraphs in graph.

        Returns:
            int: Number of disconnected subgraphs
        """
        return self._n - len(self._dfs())

    def _dfs(self, current: int = 0, visited: Set[int] = None) -> Set[int]:
        """
        Performs DFS traversal of graph.

        Args:
            current (int): Current node
            visited (Set[int]): Visited nodes

        Returns:
            Set[int]: Visited nodes
        """
        if visited is None:
            visited = set()

        visited.add(current)
        for new in range(self._n):
            if self._matrix[current][new] == GraphCell.FILLED and new not in visited:
                self._dfs(new, visited)

        return visited


class Word:
    """
    Class representing a word placed on the crossword.

    Attributes:
        _value (str): The word value
        x (int): x coordinate
        y (int): y coordinate
        direction (Direction): Word direction

    """
    _value: str

    x: int
    y: int

    _end_x: int
    _end_y: int
    direction: Direction

    def __init__(self, value: str, x: int, y: int, direction: Direction) -> None:
        """
        Constructs a word with value, coordinates and direction.

        Args:
            value (str): The word value
            x (int): x coordinate
            y (int): y coordinate
            direction (Direction): Word direction
        """
        self._value = value

        self.direction = direction

        self.x = x
        self.y = y

    def __str__(self) -> str:
        """
        String representation contains coordinates and direction.

        Returns:
            str: String representation
        """
        return f"{self.x} {self.y} {self.direction.value}"

    def __copy__(self) -> Word:
        """
        Returns copy of word.

        Returns:
            Word: Copy of word
        """
        return Word(self.value, self.x, self.y, self.direction)

    @property
    def value(self) -> str:
        """
        Gets word value (sequence of letters).

        Returns:
            str: Word value
        """
        return self._value

    @property
    def length(self) -> int:
        """
        Gets word length.

        Returns:
            int: Word length
        """
        return len(self._value)


class Crossword:
    """
    Class representing state of the crossword puzzle.

    Attributes:
        _n (int): Size n
        _m (int): Size m
        _strings (List[str]): Words to place
        fitness (int): Fitness score
        words (List[Word]): Placed words
    """

    _n: int
    _m: int
    _strings: List[str]

    fitness: int
    words: List[Word]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20) -> None:
        """
        Constructs empty crossword grid and list of words to place.

        Args:
            strings (List[str]): Words to place
            n (int, optional): Size n. Defaults to 20.
            m (int, optional): Size m. Defaults to 20.
        """
        self._strings = strings
        self._n = n
        self._m = m

        self._strings = strings

        self.words = self._generate_random_positions()
        self.fitness = self.calculate_fitness()

    def __str__(self):
        """Return a string representation of the Crossword instance."""
        return ";".join([str(x) for x in self.words])

    def __copy__(self) -> Crossword:
        """Return a shallow copy of the Crossword instance."""
        crossword = Crossword(self.strings, self.n, self.m)
        crossword.words = [copy(word) for word in self.words]
        return crossword

    @property
    def n(self) -> int:
        """Return the number of rows in the crossword grid."""
        return self._n

    @property
    def m(self) -> int:
        """Return the number of columns in the crossword grid."""
        return self._m

    @property
    def strings(self) -> List[str]:
        """Return the list of strings to be used as words in the crossword."""
        return self._strings

    def generate_safe_x_from_word(self, word: Word) -> int:
        """
        Generate a safe x-coordinate for a word placement in the crossword grid.

        Args:
            word (Word): The Word instance.

        Returns:
            int: The safe x-coordinate.
        """
        constraint = self.n - 1 - (word.length if word.direction == Direction.HORIZONTAL else 0)
        return random.randint(0, constraint)

    def generate_safe_y_from_word(self, word: Word) -> int:
        """
        Generate a safe y-coordinate for a word placement in the crossword grid.

        Args:
            word (Word): The Word instance.

        Returns:
            int: The safe y-coordinate.
        """
        constraint = self.m - 1 - (word.length if word.direction == Direction.VERTICAL else 0)
        return random.randint(0, constraint)

    def generate_safe_x_from_string(self, word: str, direction: Direction) -> int:
        """
        Generate a safe x-coordinate for a string placement in the crossword grid.

        Args:
            word (str): The string to be placed.
            direction (Direction): The direction of placement.

        Returns:
            int: The safe x-coordinate.
        """
        constraint = self.n - 1 - (len(word) if direction == Direction.HORIZONTAL else 0)
        return random.randint(0, constraint)

    def generate_safe_y_from_string(self, word: str, direction: Direction) -> int:
        """
        Generate a safe y-coordinate for a string placement in the crossword grid.

        Args:
            word (str): The string to be placed.
            direction (Direction): The direction of placement.

        Returns:
            int: The safe y-coordinate.
        """
        constraint = self.m - 1 - (len(word) if direction == Direction.VERTICAL else 0)
        return random.randint(0, constraint)

    def _generate_random_positions(self) -> List[Word]:
        """
        Generate random positions for words in the crossword grid.

        Returns:
            List[Word]: A list of Word instances with random positions.
        """
        words = []
        for word in self.strings:
            direction = random.choice(list(Direction))

            words.append(Word(
                x=self.generate_safe_x_from_string(word, direction),
                y=self.generate_safe_y_from_string(word, direction),
                direction=direction,
                value=word
            ))

        return words

    def within_bounds(self, x: int, y: int) -> bool:
        """
        Check if coordinates (x, y) are within the bounds of the crossword grid.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.

        Returns:
            bool: True if coordinates are within bounds, False otherwise.
        """
        return (0 <= x < self.n) and (0 <= y < self.m)

    def word_within_bounds(self, word: Word) -> bool:
        """
        Check if a word is entirely within the bounds of the crossword grid.

        Args:
            word (Word): The Word instance.

        Returns:
            bool: True if the word is entirely within bounds, False otherwise.
        """
        end_x = word.x + (word.length if word.direction == Direction.HORIZONTAL else 0)
        end_y = word.y + (word.length if word.direction == Direction.VERTICAL else 0)

        return self.within_bounds(word.x, word.y) and self.within_bounds(end_x, end_y)

    def print(self) -> None:
        """Prints visual representation of crossword state."""
        matrix = [[" . " for _ in range(0, self.m)] for _ in range(0, self.n)]

        for word in self.words:
            for i in range(word.length):
                new_x = word.x + (i if word.direction == Direction.HORIZONTAL else 0)
                new_y = word.y + (i if word.direction == Direction.VERTICAL else 0)
                if self.within_bounds(new_x, new_y):
                    matrix[new_x][new_y] = f" {word.value[i]} "

        print(" " + " - " * self.m + " ")
        for row in matrix:
            print(f'|{"".join(row)}|')
        print(" " + " - " * self.m + " ")
        print("\n")

    @staticmethod
    def detect_overlapping(word1: Word, word2: Word) -> bool:
        """
        Check if two words overlap in the crossword grid.

        Args:
            word1 (Word): The first Word instance.
            word2 (Word): The second Word instance.

        Returns:
            bool: True if the words overlap, False otherwise.
        """
        if word1.direction != word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            if not (abs(word1.x - word2.x) <= 1):
                return False

            if word2.y < word1.y:
                word1, word2 = word2, word1

            return word1.y <= word2.y <= word1.y + word1.length

        elif word1.direction == Direction.HORIZONTAL:
            if not abs(word1.y - word2.y) <= 1:
                return False

            if word2.x < word1.x:
                word1, word2 = word2, word1

            return word1.x <= word2.x <= word1.x + word1.length

        return False

    @staticmethod
    def check_intersection(word1: Word, word2: Word) -> bool:
        """
        Check if two words intersect in the crossword grid.

        Args:
            word1 (Word): The first Word instance.
            word2 (Word): The second Word instance.

        Returns:
            bool: True if the words intersect, False otherwise.
        """
        if word1.direction == word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            word1, word2 = word2, word1

        return word1.x <= word2.x < word1.x + word1.length and word2.y <= word1.y < word2.y + word2.length

    @staticmethod
    def check_letter_match(word1: Word, word2: Word) -> bool:
        """
        Check if corresponding letters of two intersecting words match.

        Args:
            word1 (Word): The first Word instance.
            word2 (Word): The second Word instance.

        Returns:
            bool: True if letters match, False otherwise.
        """
        if word1.direction == word2.direction:
            return False

        if word1.direction == Direction.VERTICAL:
            return word1.value[word2.y - word1.y] == word2.value[word1.x - word2.x]
        else:
            return word1.value[word2.x - word1.x] == word2.value[word1.y - word2.y]

    @staticmethod
    def check_collisions(word1: Word, word2: Word) -> int:
        """
        Check if two words collide in the crossword grid.

        Args:
            word1 (Word): The first Word instance.
            word2 (Word): The second Word instance.

        Returns:
            int: Number of collisions between the words.
        """
        if word1.direction == word2.direction:
            return 0

        collisions = 0

        if word1.direction == Direction.HORIZONTAL:
            word1, word2 = word2, word1

        collisions += (word2.y == word1.y - 1 and word2.x <= word1.x < word2.x + word2.length)

        collisions += (word2.y == word1.y + word1.length and word2.x <= word1.x < word2.x + word2.length)

        collisions += ((word2.x + word2.length - 1 == word1.x - 1) and (
                word1.y <= word2.y <= word1.y + word1.length - 1))

        collisions += ((word2.x == word1.x + 1) and (word1.y <= word2.y <= word1.y + word1.length - 1))

        return collisions

    def calculate_fitness(self) -> int:
        """Calculates fitness score for current crossword state.

        Returns:
            int: Fitness score
        """
        graph = Graph(len(self.words))
        fitness = 0

        for i in range(len(self.words)):
            word1 = self.words[i]

            fitness += 100000 * (not self.word_within_bounds(word1))

            for j in range(i + 1, len(self.words)):
                word2 = self.words[j]

                fitness += 100 * self.detect_overlapping(word1, word2)

                if self.check_intersection(word1, word2):
                    graph.fill_edge(i, j)
                    graph.fill_edge(j, i)
                    fitness += 5 * (not self.check_letter_match(word1, word2))

                fitness += 20 * self.check_collisions(word1, word2)

        fitness += graph.get_amount_of_disconnected() * 1000
        return fitness

    def generate_output(self) -> str:
        """
        Generate a string representation of the crossword solution.

        Returns:
            str: String representation of the solution.
        """
        return "\n".join(f"{str(word)}" for word in self.words)


class EvolutionaryAlgorithm:
    """
    Class implementing evolutionary algorithm for crossword solving.

    Attributes:
        _strings (List[str]): Words to place
        _n (int): Size n
        _m (int): Size m
        population (List[Crossword]): Population of solutions
    """

    _strings: List[str]

    _n: int
    _m: int
    _population_size: int

    population: List[Crossword]

    def __init__(self, strings: List[str], n: int = 20, m: int = 20, population_size: int = 100) -> None:
        """
        Constructs EA instance with parameters.

        Args:
            strings (List[str]): Words to place
            n (int, optional): Size n. Defaults to 20.
            m (int, optional): Size m. Defaults to 20.
            population_size (int, optional): Population size. Defaults to 100.
        """
        self._strings = strings
        self._n = n
        self._m = m
        self._population_size = population_size

        self.population = []

    @property
    def strings(self) -> List[str]:
        """
        Get the list of strings to be used in crosswords.

        Returns:
            List[str]: The list of strings.
        """
        return self._strings

    @property
    def population_size(self) -> int:
        """
        Get the size of the population.

        Returns:
            int: The population size.
        """
        return self._population_size

    @property
    def n(self) -> int:
        """
        Get the number of rows in the crossword grid.

        Returns:
            int: The number of rows.
        """
        return self._n

    @property
    def m(self) -> int:
        """
        Get the number of columns in the crossword grid.

        Returns:
            int: The number of columns.
        """
        return self._m

    def calculate_fitnesses(self) -> None:
        """
        Calculate the fitness for each crossword in the population."""
        for crossword in self.population:
            crossword.fitness = crossword.calculate_fitness()

    def generate_random_population(self) -> List[Crossword]:
        """
        Generate a random population of crosswords.

        Returns:
            List[Crossword]: The generated population.
        """
        return [Crossword(self.strings, self.m, self.n) for _ in range(self.population_size)]

    @staticmethod
    def _select_best(initial_population: List[Crossword], best_individuals_percentage=10) -> List[Crossword]:
        """
        Select the best individuals from the population.

        Args:
            initial_population (List[Crossword]): The initial population.
            best_individuals_percentage (int): Percentage of the best individuals to select.

        Returns:
            List[Crossword]: The selected best individuals.
        """
        population = [copy(crossword) for crossword in initial_population]
        best_individuals_length = len(population) // best_individuals_percentage

        return population[:best_individuals_length:]

    @staticmethod
    def _select_rest(initial_population: List[Crossword], best_individuals_percentage=10) -> List[Crossword]:
        """
        Select the (len(population) - best_individuals) of the individuals from the population.

        Args:
            initial_population (List[Crossword]): The initial population.
            best_individuals_percentage (int): Percentage of the best individuals to select.

        Returns:
            List[Crossword]: The selected rest of the individuals.
        """
        population = [copy(crossword) for crossword in initial_population]
        rest_individuals_length = len(population) - (len(population) // best_individuals_percentage)

        rest_individuals = EvolutionaryAlgorithm._roulette_selection(population, rest_individuals_length)

        return [
            EvolutionaryAlgorithm._crossover(
                EvolutionaryAlgorithm._tournament_selection(rest_individuals),
                EvolutionaryAlgorithm._tournament_selection(rest_individuals)
            )
            for _ in range(len(rest_individuals))
        ]

    @staticmethod
    def _roulette_selection(initial_population: List[Crossword], k: int = 1) -> List[Crossword]:
        """
        Perform roulette selection to choose individuals based on their fitness.

        Args:
            initial_population (List[Crossword]): The initial population.
            k (int): Number of individuals to select.

        Returns:
            List[Crossword]: The selected individuals.
        """
        population = [copy(crossword) for crossword in initial_population]
        return random.choices(
            population=population,
            weights=[x.fitness for x in population],
            k=k
        )

    @staticmethod
    def _tournament_selection(population: List[Crossword], tournament_size=3) -> Crossword:
        """
        Perform tournament selection to choose individuals based on their fitness.

        Args:
            population (List[Crossword]): The population.
            tournament_size (int): Size of the tournament.

        Returns:
            Crossword: The selected individual.
        """
        return min(random.sample(population, k=tournament_size), key=lambda x: x.fitness)

    def _selection(self, initial_population: List[Crossword]) -> List[Crossword]:
        """
        Perform selection to create the next generation of individuals.

        Args:
            initial_population (List[Crossword]): The initial population.

        Returns:
            List[Crossword]: The selected individuals for the next generation.
        """

        population = [copy(crossword) for crossword in initial_population]

        return self._select_best(population) + self._mutate_population(self._select_rest(population))

    @staticmethod
    def _crossover(parent1: Crossword, parent2: Crossword, crossover_rate: float = 0.5) -> Crossword:
        """
        Perform crossover between two parents to create a child.

        Args:
            parent1 (Crossword): The first parent.
            parent2 (Crossword): The second parent.
            crossover_rate (float): Crossover rate.

        Returns:
            Crossword: The created child.
        """
        child1 = copy(parent1)
        child2 = copy(parent2)

        for i in range(len(parent1.words)):
            if random.random() < crossover_rate:
                child1.words[i].x = parent2.words[i].x
                child1.words[i].y = parent2.words[i].y
                child1.words[i].direction = parent2.words[i].direction

                child2.words[i].x = parent1.words[i].x
                child2.words[i].y = parent1.words[i].y
                child2.words[i].direction = parent1.words[i].direction

        return random.choice([child1, child2])

    def _mutate_population(self, initial_population: List[Crossword], mutation_rate: float = 0.5) -> List[Crossword]:
        """
        Perform mutation on the population.

        Args:
            initial_population (List[Crossword]): The initial population.
            mutation_rate (float): Mutation rate.

        Returns:
            List[Crossword]: The mutated population.
        """
        return [self._mutate(x, mutation_rate) for x in [copy(crossword) for crossword in initial_population]]

    @staticmethod
    def _mutate(initial_individual: Crossword, mutation_rate: float = 0.5) -> Crossword:
        """
        Perform mutation on an individual.

        Args:
            initial_individual (Crossword): The initial individual.
            mutation_rate (float): Mutation rate.

        Returns:
            Crossword: The mutated individual.
        """
        individual = copy(initial_individual)

        for word in individual.words:
            if random.random() < mutation_rate:
                mutation_probability = random.random()

                # change only x
                if mutation_probability < 0.17:
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))

                # change only y
                elif mutation_probability < 0.34:
                    word.y = random.randint(0, individual.generate_safe_y_from_word(word))

                # change both x and y
                elif mutation_probability < 0.5:
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))
                    word.y = random.randint(0, individual.generate_safe_y_from_word(word))

                # change only direction
                elif mutation_probability < 0.625:
                    word.direction = random.choice(list(Direction))

                # change direction and x
                elif mutation_probability < 0.75:
                    word.direction = random.choice(list(Direction))
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))

                # change direction and y
                elif mutation_probability < 0.875:
                    word.direction = random.choice(list(Direction))
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))

                # change direction and both x and y
                elif mutation_probability < 1:
                    word.direction = random.choice(list(Direction))
                    word.x = random.randint(0, individual.generate_safe_x_from_word(word))
                    word.y = random.randint(0, individual.generate_safe_y_from_word(word))

        return individual

    def run(self, name: str, max_generation=100000, current_generation: int = 0, current_try: int = 0,
            max_tries: int = 100) -> Tuple[float, int, Crossword]:
        """Runs evolutionary algorithm to solve crossword.

        Args:
            name (str): Run name
            max_generation (int, optional): Max generations. Defaults to 100000.
            current_generation (int, optional): Current generation. Defaults to 0.
            current_try (int, optional): Current try. Defaults to 0.
            max_tries (int, optional): Max tries. Defaults to 100.

        Returns:
            Crossword: Best solution
        """

        idle_generations = 0
        max_fitness = MAX_INT

        self.population = self.generate_random_population()

        start_time = time.time()

        for generation in range(current_generation, max_generation):

            self.calculate_fitnesses()

            self.population = sorted(self.population, key=lambda x: x.fitness)

            if self.population[0].fitness == 0:
                break

            if self.population[0].fitness == max_fitness:
                idle_generations += 1
            elif self.population[0].fitness < max_fitness:
                max_fitness = self.population[0].fitness
                idle_generations = 0

            print(name)
            print(f"Generation: {generation} and {current_try}'th try")
            print(f"Best fitness: {self.population[0].fitness}")
            self.population[0].print()

            if idle_generations >= 1000 * len(self.strings):
                return self.run(
                    name=name,
                    max_generation=max_generation,
                    current_generation=0,
                    current_try=current_try + 1,
                    max_tries=max_tries
                )
                break

            self.population = self._selection(self.population)

        if current_try == 0:
            print("Best:")
            self.population[0].print()

        return (time.time() - start_time) / 60, current_generation, self.population[0]


class Assignment:
    _base_directory: str

    def __init__(self, base_directory: str = ".") -> None:
        """
        Initialize an Assignment instance.

        Args:
            base_directory (str): The base directory for inputs, outputs, and images.
        """
        self._base_directory = base_directory

    @property
    def base_directory(self) -> str:
        """
        Get the base directory.

        Returns:
            str: The base directory.
        """
        return self._base_directory

    @property
    def inputs_folder(self) -> str:
        """
        Get the inputs folder path.

        Returns:
            str: The inputs folder path.
        """
        return f"{self.base_directory}/inputs"

    @property
    def outputs_folder(self) -> str:
        """
        Get the outputs folder path.

        Returns:
            str: The outputs folder path.
        """
        return f"{self.base_directory}/outputs"

    @property
    def images_folder(self) -> str:
        """
        Get the images folder path.

        Returns:
            str: The images folder path.
        """
        return f"{self.base_directory}/images"

    def read_input(self) -> List[Tuple[str, int, List[str]]]:
        """
        Read input test cases from input files.

        Returns:
            List[Tuple[str, List[str]]]: List of test cases.
        """
        tests = []
        for test in filter(lambda x: x.startswith("input") and x.endswith(".txt"), os.listdir(self.inputs_folder)):
            with open(f"{self.inputs_folder}/{test}") as file:
                data = file.readlines()
                tests.append((test, len(data), [x.rstrip("\n") for x in data]))

        return tests

    def solve(self) -> None:
        """Solve the crossword assignment problem for all input test cases."""
        for test in self.read_input():
            name = test[0]
            length = test[1]
            dataset = test[2]

            logging.info(f"{time.time()}: Started checking {name}")

            crossword = EvolutionaryAlgorithm(dataset)

            execution_time, generation, answer = crossword.run(name)

            logging.info(f"{time.time()}: Ended checking {name}")

            with open(f"{self.outputs_folder}/output{name[5::]}", mode="w") as file:
                file.write(answer.generate_output())

            with open(f"{self.base_directory}/statistics.csv", mode="a") as file:
                file.write(f"{length};{execution_time};{generation}\n")


def main() -> None:
    """Main function to execute the crossword assignment problem."""
    assignment = Assignment()

    assignment.solve()


if __name__ == "__main__":
    main()
