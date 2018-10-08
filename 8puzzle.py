import numpy as np
import random
from collections import deque

class Puzzle:
    """Create any kind of puzzle using Numpy"""
    def __init__(self, number):
        self.number = number
        self.matrix = np.zeros((number,number), dtype=int)
        self.target_state = np.zeros((number,number), dtype=int)
    def _generator(self, kind=None):
        """Generator to generate a random puzzle"""
        from random import sample
        list_numbers = sample(range(0, (self.number**2)), self.number**2)
        if kind == 'sorted':
            list_numbers = sorted(sample(range(0, (self.number**2)), self.number**2))
        for i in list_numbers:
            yield i
    def generate_puzzle(self):
        """Generate the initial state"""
        n = self._generator()
        for i in self.matrix:
            for j in range(len(i)):
                i[j] = next(n)
    def generate_puzzle_target(self):
        """Generate the target state"""
        n = self._generator(kind='sorted')
        for i in self.target_state:
            for j in range(len(i)):
                i[j] = next(n)

def valid_moves(state):
    """Return possible moves for a state 3x3"""
    k, l = np.where(state==0) # the zero location
    k, l = k[0], l[0]
    moves = []
    if k == 0:
        if l == 0:
            moves = ['right', 'down']
        elif l == 1:
            moves = ['left', 'right', 'down']
        elif l == 2:
            moves = ['left', 'down']
    elif k == 1:
        if l == 0:
            moves = ['up', 'right', 'down']
        elif l == 1:
            moves = ['left', 'up', 'right', 'down']
        elif l == 2:
            moves = ['left', 'up', 'down']
    elif k == 2:
        if l == 0:
            moves = ['up', 'right']
        elif l == 1:
            moves = ['left', 'up', 'right']
        elif l == 2:
            moves = ['left', 'up']
    return moves

def apply(state, move):
    """Return the result of a move"""
    k, l = np.where(state==0)
    k, l = k[0], l[0]
    new_state = np.array(state)
    if k == 0:
        if l == 0:
            if move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
        elif l == 1:
            if move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
            elif move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
        elif l == 2:
            if move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
    elif k == 1:
        if l == 0:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
        elif l == 1:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
            elif move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
        elif l == 2:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
            elif move == 'down':
                new_state[k][l] = new_state[k+1][l]
                new_state[k+1][l] = 0
    elif k == 2:
        if l == 0:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
        elif l == 1:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
            elif move == 'right':
                new_state[k][l] = new_state[k][l+1]
                new_state[k][l+1] = 0
        elif l == 2:
            if move == 'up':
                new_state[k][l] = new_state[k-1][l]
                new_state[k-1][l] = 0
            elif move == 'left':
                new_state[k][l] = new_state[k][l-1]
                new_state[k][l-1] = 0
    return new_state

class Queue:
    """LIFO Queue"""
    def __init__(self):
        self.visits = deque()
    def put(self, value):
        self.visits.append(value)
    def get_all(self):
        return list(self.visits)
    def get(self):
        return self.visits.pop()
    def empty(self):
        return len(self.visits) == 0
def build(matrix, target):
    """Find the path and return result"""
    goal = [i[j] for i in target for j in range(len(i))] # the result expected
    queue = Queue()
    initial = [i[j] for i in np.array(matrix) for j in range(len(i))] 
    queue.put(initial)
    while not queue.empty():
        # Not too optimal but can find the path
        current = queue.get() # get the last move
        if current == goal:
            return target
        new_states = [apply(matrix, move) for move in valid_moves(matrix)]
        n = random.choice(new_states)
        matrix = np.array(n)#upadte the matrix
        n = [i[j] for i in n for j in range(len(i))]
        queue.put(n) # add the chosen move to the queue
            
        
puzzle8 = Puzzle(3)
puzzle8.generate_puzzle()
puzzle8.generate_puzzle_target()
print('----initial')
print(puzzle8.matrix)
print('----target')
print(puzzle8.target_state)
#state = np.array([[8, 1, 2],[4, 0, 6],[7, 5, 3]])
#new_states = [apply(state, move) for move in valid_moves(state)]
print('----final')
print(build(puzzle8.matrix, puzzle8.target_state))