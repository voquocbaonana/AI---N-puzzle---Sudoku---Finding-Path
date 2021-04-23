# Depth first search for N puzzles problem
import random
import copy
import time
# command to move empty block
LEFT = 'Left'
UP = 'Up'
RIGHT = 'Right'
DOWN = 'Down'
MOVES = [UP, DOWN, LEFT, RIGHT]
# -------------------


class Node:
    # Each move is saved as class named Node.
    def __init__(self, state, parent, action, depth):
        # parameter:    state: current state
        #               parent: state of parent
        #               action: move of father node (left/right/up/down)
        #               depth: depth of current node
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def solution(self):
        # back tracking to find the solution string when reach the goals
        # return: List of command to reach the goal from the inital state to the goal
        node, path = self, []
        while node:
            path.append(node.action)
            node = node.parent
        # remove last element. Because assign 0 for inital state
        return list(reversed(path[:-1]))

    def expand(self):
        # expand current node by swapping empty block to right/left/up/down.
        # return: list of Nodes by moving empty block to right/left/up/down
        lst = []
        for i in MOVES:
            temp = move(self.state, i)
            if temp:
                lst.append(Node(temp, self, i, self.depth+1))
        return lst


def move(state, command):
    # Move empty block to right,left,up,down by command
    # parameter:    state: current state of the board
    #               command: command to move empty block to left/right/left/down
    # return:       new state if can swap else return None
    state_copy = state[:]
    index = state.index(0)
    length = len(state)  # N*N
    size = int(length**0.5)  # N
    if command == UP:
        if index >= size:
            state_copy[index], state_copy[index -
                                          size] = state_copy[index-size], state_copy[index]
        else:
            return None
    elif command == DOWN:
        if index < length-size:
            state_copy[index], state_copy[index +
                                          size] = state_copy[index+size], state_copy[index]
        else:
            return None
    elif command == LEFT:
        if index not in range(0, length, size):
            state_copy[index], state_copy[index -
                                          1] = state_copy[index-1], state_copy[index]
        else:
            return None
    elif command == RIGHT:
        if index not in range(size-1, length, size):
            state_copy[index], state_copy[index +
                                          1] = state_copy[index+1], state_copy[index]
        else:
            return None
    return state_copy


def createBoard(N):
    # gernegate inital board by N. Example if N=3. generate 3x3 board
    # parameter:    N: numbers of puzzle in one row
    # return:       lits of NxN elements with random orders.
    # Use 1D arrays instead of matrix to improve performance
    return random.sample(range(0, N*N), N*N)


def dfs(initalNode):

    # generate goal: if N=3 goal: 1 2 3 4 5 6 7 8 0
    goal = [i for i in range(1, len(initalNode))]
    goal.append(0)
    # ---------------------------------------------------
    # create stack
    stack = list()
    # create list of visited to avoid endless loops
    visited = list()
    stack.append(Node(initalNode, None, None, 0))
    while stack:
        node = stack.pop()
        visited.append(node.state)
        if node.state == goal:
            return node
        else:
            neighbors = node.expand()
            for i in neighbors:
                # Only check state in visited and no     if state exists in stack to improve performance
                if i.state not in visited:
                    stack.append(i)
    print('Can not find solution !')
    return None


def printBoard(lst):
    # print list as matrix
    N = int(len(lst)**0.5)
    print('-'*5)
    for k in [lst[i:i+N] for i in range(0, len(lst), N)]:
        print(k)
    print('-'*5)


def main():
    N2 = int(input("Numbers of puzzle:")) 
    N= int((N2+1)**0.5) # sqrt(N**2)
    inital = createBoard(N)
    printBoard(inital)
    print("Solving...")
    startTime = time.time()
    node = dfs(inital)
    print("Solved !")
    timeExecuted = time.time()-startTime
    file = open('result.txt', 'w')
    file.write("Inital state:"+str(inital))
    file.write("\nPath to goal: " + str(node.solution()))
    file.write("\nCost of path: " + str(len(node.solution())))
    file.write("\nSearch depth: " + str(node.depth))
    file.write("\nTime to executed: {0:.4f}s".format(timeExecuted))


if __name__ == "__main__":
    main()
    #inital = [1, 2, 3, 4, 5, 0,6,7, 8]
