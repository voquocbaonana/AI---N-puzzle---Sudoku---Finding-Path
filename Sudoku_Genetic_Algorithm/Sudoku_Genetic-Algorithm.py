import numpy as np
import random as rd
from collections import Counter
from copy import deepcopy
from random import sample
import time
from past.builtins import xrange
from past.builtins import  range
rd.seed()

class Flag:
    # create flag for each cell. This class save location of cell (x,y) and list of 9 flags
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.flag = [1]*9

def setFlagZero(flags, idx, value):
    # parameters:    flags: list of Flags
    #               idx: index of cell has value
    #               value: set 0 for flag has that value
    for i in range(len(flags)):
        if i != idx:
            flags[i].flag[
                value - 1] = 0  # value -1 because sudoku has values in range(1,10) while index have value range(0,9)

def getRemovalFlag(board):
    removal_flag = deepcopy(board)
    for i in range(len(removal_flag)):
        for j in range(len(removal_flag[i])):
            if removal_flag[i][j] == 0:
                removal_flag[i][j] = 1
            else:
                removal_flag[i][j] = 0
    return removal_flag

def getChildBoard(board, row, col):
    # get 3x3 child matrix
    box_list = []
    N = int(len(board)**0.5)
    # 0 1 2  3 4 5  6 7 8
    mul_row = row // N
    mul_col = col // N  # 1
    for i in range(0, N):
        for j in range(0, N):
            row_idx = N*mul_row + i
            col_idx = N*mul_col + j
            box_list.append(board[row_idx][col_idx])

    return box_list

def predeterminedCell(board, removal_flag):
    #parameters:    board: current board sudoku NxN
    #               removal_flag: NxN matrix displays the positions of emtpy cells
    #return:    new board with predetermined cell and the updated removal_flags
    N = len(board)
    #create Flags for each cells
    flags = [[Flag(x, y) for y in range(N)] for x in range(N)]

    flags = np.array(flags)
    for row in range(N):
        for col in range(N):
            #get value of this position
            value = board[row][col]
            if value != 0:
                #set flag 0 of this value to other cells on the same column and row
                setFlagZero(flags[row, :], col, board[row][col])
                setFlagZero(flags[:, col], row, board[row][col])
    for i in flags:
        for j in i:
            # value=index+1
            valuePredetermined = j.flag.index(1)+1
            #if flags of cell has only 1 flag (value 1) or sum of list equal to 1. Cell only has that value
            # check if this value not in miniBoard (3x3) of this cell and if this cell is empty
            if sum(j.flag) == 1 and board[j.X][j.Y] == 0 and valuePredetermined not in getChildBoard(board, j.X, j.Y):
                #update value for empty cell
                board[j.X][j.Y] = valuePredetermined
                #delete this empty cell
                removal_flag[j.X][j.Y] = 0
    return board, removal_flag

def generateBoard(n):
    """
        Function create board with numbers of empty which user can modified by numbers_empty
        Input: n (3)
        Output: Input board
    """
    # n: the dimension of the interior boxes, the board is n^2 * n^2
    side = n*n
    nums = sample(range(1, side + 1), side)  # random numbers

    board_result = [[nums[(n * (r % n) + r // n + c) % side]
                     for c in range(side)] for r in range(side)]

    board_removal = deepcopy(board_result)

    # create flag table. if removal value, flag will be set to 1
    removal_map = [[0 for _ in range(side)] for _ in range(side)]
    numbers_empty = side*side//2 + 8       #Numbers of zero in matrix ( empty in board)
    for i in sample(range(side*side), numbers_empty):
        board_removal[i//side][i % side] = 0
        removal_map[i//side][i % side] = 1
    return board_result, board_removal, removal_map

def is_used(board, row, col, val):
    """ This function is used to check if a value is suitable 
    for that cell (check duplicate in its row, its column and its block) 
    If it is used already, this func returns True, else False
    """
    c = col // 3    # get the integer index of the first column of block
    r = row // 3    # get the integer index of the first row of block
    # get numbers from the board to block
    block = [y[c*3:c*3+3] for y in board[r*3:r*3+3]]
    # get numbers from the board to block
    block = [x for y in block for x in y]
    if val in board[row] or val in board[:, col] or val in block:
        return True
    return False


class Candidate:
    def __init__(self):
        self.value = np.zeros((9, 9), dtype=int)
        self.fitness = None
        return

    def fitness_function(self):
        """ duplicates are counted due to the times a number appears in a row/column/block
        if it appears 1 time than duplicate = 0, if it appears n time (n>1) then duplicates += n
        fitness = num_of_duplicates/243 (where 243 is the maximum duplicates on a board, which means the board has 81 duplicated number)
        So the bigger the fitness is, the less chance for it to be chosen (because of the number of duplicates is big, the result will
        be further from the real result
        If the fitness is closer to 0, then the result will be better)
        """
        count_row = 0
        count_col = 0
        count_block = 0
        # count duplicates in rows
        for i in range(0, 9):
            cnt = Counter(self.value[i])
            for j in range(1, 10):
                if cnt[j] != 1:
                    count_row += cnt[j]
        # count duplicates in columns
        for i in range(0, 9):
            cnt = Counter(self.value[:, i])
            for j in range(1, 10):
                if cnt[j] != 1:
                    count_col += cnt[j]
        # create blocks from board
        block = []
        for i in range(0, 3):
            row = np.zeros(9, dtype=int)
            for j in range(0, 3):
                row = [y[j*3:j*3+3] for y in self.value[i*3:i*3+3]]
                row = [x for y in row for x in y]
                block.append(row)
        # count duplicates in blocks
        for i in range(0, 9):
            cnt = Counter(block[i])
            for j in range(1, 10):
                if cnt[j] != 1:
                    count_block += cnt[j]
        fitness = (count_row + count_col + count_block)/243
        self.fitness = fitness
        return

    def mutate(self, original, mutation_rate):
        """ 
        Ta dùng hàm này để tạo "đột biến", tức là đổi chỗ 2 số bất kì trong một hàng để tạo con mới từ cha của nó
        - Input: Bảng sudoku cần tạo đột biến, tỉ lệ đột biến    
        - Output: Bảng sudoku đã đột biến
        """

        r = rd.uniform(0, 1)  # create a random real number between 0 and 1
        swapped = False

        if r < mutation_rate:  # if random number is less than mutation_rate, returns
            while not swapped:
                row = rd.randint(0, 8)
                col1 = rd.randint(0, 8)
                col2 = rd.randint(0, 8)
                while col2 == col1:
                    col2 = rd.randint(0, 8)
                if original[row][col1] == 0 and original[row][col2] == 0:
                    if not is_used(original, row, col1, self.value[row][col1]) and not is_used(original, row, col2,
                                                                                               self.value[row][col2]):
                        temp = self.value[row][col1]
                        self.value[row][col1] = self.value[row][col2]
                        self.value[row][col2] = temp
                        swapped = True
        return swapped

    def resetting_mutation(self, original, mutation_rate):
        """
                This func is used to "mutate" by pick a random row and then pick a random column, replace the value at that position with
                a random number from 1 to 9
                - Input: An original board, mutation rate
                - Output: A board which is mutated
        """
        r = rd.uniform(0, 1)
        reset = False
        if r < mutation_rate:
            while not reset:
                row = rd.randint(0, 8)
                col = rd.randint(0, 8)
                if original[row][col] == 0:
                    self.value[row][col] = rd.randint(1, 9)
                    reset = True
        return reset

class Population:
    def __init__(self):
        self.candidates = []
        return

    def generate(self, N, given):
        """
        Generate a new population with N candidates
        - Input:
            + N: Number of candidates
            + given: The seed to generate new candidates
        """
        self.candidates = []

        # checker is used to help us determine what numbers can be filled to each cell of the sudoku board
        checker = Candidate()
        checker.value = [[[] for j in range(0, 9)] for i in range(0, 9)]
        for row in range(0, 9):
            for column in range(0, 9):
                for value in range(1, 10):
                    if ((given[row][column] == 0) and not (is_used(given, row, column, value))):
                        # Value is available
                        checker.value[row][column].append(value)
                    elif given[row][column] != 0:
                        # Given/known value from orignal board
                        checker.value[row][column].append(given[row][column])
                        break

        for _ in range(N):
            child = Candidate()
            for i in range(0, 9):
                for j in range(0, 9):
                    if given[i][j] != 0:
                        child.value[i][j] = given[i][j]
                    else:
                        # Add value from checker to new candidate
                        child.value[i][j] = checker.value[i][j][rd.randint(
                            0, len(checker.value[i][j]) - 1)]

                    # If we didn't make a valid board, we retry until 500,000 times
                    loop = 0
                    while len(set(child.value[i])) != 9:
                        loop += 1
                        if loop > 500000:
                            return 0
                        for j in range(9):
                            child.value[i][j] = checker.value[i][j][rd.randint(
                                0, len(checker.value[i][j]) - 1)]
            self.candidates.append(child)
        self.fitness_function()
        return 1

    def fitness_function(self):
        """
        Update fitness for every candidate in candidates list
        - Input: None
        - Output: None
        """
        for i in self.candidates:
            i.fitness_function()
        return

    def ranking(self):
        """
        To rank candidates in list by its fitness
        - Input: None
        - Output: None
        """
        self.candidates.sort(key=lambda x: x.fitness)

class Fixed(Candidate):
    def __init__(self, value):
        self.value = value
        return

    def has_duplicate(self):
        """
        To check if a row/column/block has duplicates
        - Input: None
        - Output: None
        """
        for row in range(0, 9):
            for col in range(0, 9):
                if self.value[row][col] != 0:
                    count_row = (list(self.value[row])).count(
                        self.value[row][col])
                    count_column = (list(self.value[:, col])).count(
                        self.value[row][col])
                    c = col // 3
                    r = row // 3
                    block = [y[c*3:c*3+3] for y in self.value[r*3:r*3+3]]
                    block = [int(x) for y in block for x in y]
                    count_block = block.count(self.value[row][col])

                    if count_row > 1 or count_column > 1 or count_block > 1:
                        return True
        return False


class CycleCrossover(object):
    def __init__(self):

        return

    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()

        # Make a copy of the parent genes.
        child1.value = np.copy(parent1.value)
        child2.value = np.copy(parent2.value)

        r = rd.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = rd.uniform(0, 1.1)

        # Perform crossover.
        if (r > crossover_rate):
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = rd.randint(0, 8)
            crossover_point2 = rd.randint(1, 9)
            while (crossover_point1 == crossover_point2):
                crossover_point1 = rd.randint(0, 8)
                crossover_point2 = rd.randint(1, 9)

            if (crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1,crossover_point2):

                child1.value[i], child2.value[i] = self.crossover_rows(
                child1.value[i], child2.value[i])
        return child1, child2

    def crossover_rows(self, row1, row2):
        """
            Insert values in parent to childs
            If the ith value of parent 1 not same parent 2 then continue browsing
            Input: ith row in parent 1 and ith row in parent 2
            Output: 2 rows new child which combined from 2 parent row
        """
        child_row1 = np.zeros(9)  #child1 row
        child_row2 = np.zeros(9)  #child2 row
        remaining = range(1, 9 + 1)
        cycle = 0

        while ((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if (cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                prevalue = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                nextvalue = row2[index]

                while (nextvalue != prevalue):  # While cycle not done...
                    index = self.find_value(row1, nextvalue) #where index of nextvalue
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    nextvalue = row2[index]
                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                prevalue = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                nextvalue = row2[index]

                while (nextvalue != prevalue):  # While cycle not done...
                    index = self.find_value(row1, nextvalue) #where index of nextvalue
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    nextvalue = row2[index]

                cycle += 1

        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if (parent_row[i] in remaining):
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if (parent_row[i] == value):
                return i

class Tournament(object):
    """ The crossover function requires two parents to be selected from the population pool. The Tournament class is used to do this.

    Two individuals are selected from the population pool and a random number in [0, 1] is chosen. If this number is less than the 'selection rate' (e.g. 0.2), then the fitter individual is selected; otherwise, the weaker one is selected.
    """

    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[rd.randint(0, len(candidates) - 1)]  # parent 1
        c2 = candidates[rd.randint(0, len(candidates) - 1)]  # parent 2
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if (f1 < f2):
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        # selection_rate = 0.2
        selection_rate = 0.2
        r = rd.uniform(0, 1.1)
        while (r > 1):  # Outside [0, 1] boundary. Choose another.
            r = rd.uniform(0, 1.1)
        if (r > selection_rate):
            return fittest
        else:
            return weakest


class Sudoku():
    def __init__(self):
        self.original = None
        return

    def loadData(self, value):
        self.original = Fixed(value)
        return

    def solve(self):
        Number_can = 1000  # Number of candidates (i.e. population size).
        Number_elit = int(0.05 * Number_can)  # Number of elites.
        Number_ger = 10000  # Number of generations.

        # Mutation parameters.
        mutation_rate = 0.06

        if self.original.has_duplicate() == True:
            return
        #Create Initial Board
        removal_flag = getRemovalFlag(self.original.value)
        board, removal_flag = predeterminedCell(self.original.value, removal_flag)

        self.population = Population()
        if self.population.generate(Number_can, board) == 1:
            pass
        else:
            return (-1, 1)

        for generation in range(0, Number_ger):  # inform fitness step by step
            # Check for a solution.
            best_fitness = 1.0
            for c in range(0, Number_can):
                fitness = self.population.candidates[c].fitness

                if (fitness == 0.0):  # Goal
                    print("Solution found at generation %d!" % generation)
                    print(self.population.candidates[c].value)
                    return (generation, self.population.candidates[c])
                # Find the best fitness and corresponding chromosome
                if (fitness < best_fitness):
                    best_fitness = fitness

            print("Generation:", generation, " Best fitness:", best_fitness)

            # Create the next population.
            next_population = []

            # Select elites (the fittest candidates) and preserve them for the next generation.
            self.population.ranking()
            elites = []
            for e in range(0, Number_elit):
                elite = Candidate()
                elite.value = np.copy(self.population.candidates[e].value)
                elites.append(elite)

            # Create the rest of the candidates.

            for count in range(Number_elit, Number_can, 2):
                # Select parents from population via a tournament.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                # Crossover.
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=0.0)

                # Mutate child1.
                child1.fitness_function()
                child1.mutate(board, mutation_rate)
                child1.fitness_function()


                # Mutate child2.
                child2.fitness_function()
                child2.mutate(board, mutation_rate)
                child2.fitness_function()

                # Add children to new population.
                next_population.append(child1)
                next_population.append(child2)

            # Append elites onto the end of the population. These will not have been affected by crossover or mutation.
            for e in range(0, Number_elit):
                next_population.append(elites[e])

            # Select next generation.
            self.population.candidates = next_population
            self.population.fitness_function()

        print("No solution found.")
        return (-2, 1)


def exist(board, val):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return True
    return False

s = Sudoku()
board_result, board_removal, removal_flag = generateBoard(3)
s.loadData(np.array(board_removal).reshape(9,9).astype(int))
print(s.original.value)
start_time = time.time()
generation, solution = s.solve()

time_elapsed = '{0:6.2f}'.format(time.time()-start_time)
str_print = "Solution found at generation: " + str(generation) + \
                        "\n" + "Time elapsed: " + str(time_elapsed) + "s"
print(str_print)