import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod

COLOR_START = 20
COLOR_GOAL = 30
COLOR_WALL = 10
COLOR_PATH = 40
COLOR_GHOST = 50

COST_GHOST = 1000
MIN_DISTANCE = 5

MOVES = [(0,0), (0,1), (0,-1), (-1,0), (1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
UNIFORM = np.ones(len(MOVES))/len(MOVES)

ACTIONS = [(' S', 1, (0, 1)),
           (' N', 1, (0,-1)),
           (' W', 1, (-1, 0)),
           (' E', 1, (1, 0)),
           ('SE', 1.414, (1, 1)),
           ('NE', 1.414, (1,-1)),
           ('SW', 1.414, (-1, 1)),
           ('NW', 1.414, (-1, -1))
           ] 

def shift(state, vector):
    x1,y1,x2,y2 = state
    dx1,dy1,dx2,dy2 = vector
    return (x1+dx1,y1+dy1,x2+dx2,y2+dy2)

def trunc(x, bound):
    if x<0:
        return 0
    elif x>bound:
        return bound
    else:
        return x

def l2_distance(p1, p2):
    d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return d
    
class Action:
    def __init__(self, name, cost, effect, distribution):
        self.name = name
        self.distribution = np.array(distribution)
        self.costs = cost*np.ones(len(self.distribution))
        self.effects = [effect+move for move in MOVES]

    def __repr__(self):
        return self.name
    
    def successors(self, state):
        return [shift(state,vector) for vector in self.effects]
    
    def sample(self, state):
        samples = np.random.choice(range(len(self.effects)), size=1, p=self.distribution)
        return shift(state,self.effects[samples[0]]), self.costs[samples[0]]

class Environment:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.map = np.zeros((height, width))
        self.actions = [Action(name, cost, effect, UNIFORM) for (name, cost, effect) in ACTIONS]
        self.start = (0, self.height-1, self.width//2, self.height//2)
        self.goals = [(self.width-1, 0)]

    def generate_map(self):
        # self.map = COLOR_WALL*np.random.binomial(1,0.05,(self.height, self.width))
        # self.map[(2*self.height) // 3, 0:self.width-3] = COLOR_WALL
        # self.map[self.width // 3, 3:] = COLOR_WALL
        # self.map[2*self.height // 3, 0:-4] = COLOR_WALL
        # self.map[2*self.height // 3, -3:] = 0
        self.map[self.start[1], self.start[0]] = COLOR_START
        self.map[self.start[3], self.start[2]] = COLOR_GHOST
        for x,y in self.goals:
            self.map[y, x] = COLOR_GOAL

    def display(self):
        plt.imshow(self.map)
        plt.show()

    def display_path(self, path):
        m = np.copy(self.map)
        for i, state in enumerate(path):
            if self.is_inside(state):
                m[state[1], state[0]] = i*COLOR_PATH
        plt.figure(1)
        plt.imshow(m)

        m = np.copy(self.map)
        for i, state in enumerate(path):
            if self.is_inside(state):
                m[state[3], state[2]] = i*COLOR_GHOST
        plt.figure(2)
        plt.imshow(m)
        plt.show()

    def is_inside(self, state):
        out = True
        for i in range(len(state)//2):
            if state[2*i] < 0 or state[2*i] >= self.width:
                out = False
            if state[2*i+1] < 0 or state[2*i+1] >= self.height:
                out = False
        return out
    
    def project(self, state):
        if self.is_inside(state):
            return state
        else:
            x1,y1,x2,y2 = state
            return (trunc(x1, self.width-1),
                    trunc(y1, self.height-1),
                    trunc(x2, self.width-1),
                    trunc(y2, self.height-1))

    # def is_wall(self, state):
    #     return self.map[state.y, state.x] == COLOR_WALL

    def is_goal(self, state):
        return state[0] == self.goals[0][0] and state[1] == self.goals[0][1]

    def is_terminal(self, state):
        return self.is_goal(state) or l2_distance((state[0], state[1]), (state[2], state[3]))<MIN_DISTANCE 
        
    def get_terminal_cost(self, state):
        if self.is_goal(state):
            return 0
        else:
            return COST_GHOST
        
    def get_applicable(self, state):
        if self.is_terminal(state):
            return []
        else:
            actions = self.actions
            return actions

    def get_successors(self, state, action):
        return [self.project(s) for s in action.successors(state)]
    
    def get_sampled_successor(self, state, action):
        t, cost = action.sample(state)
        return self.project(t), cost
    
    def heur(self, state):
        h = np.inf
        for g in self.goals:
            d = np.sqrt((state[0]-g[0])**2 + (state[1]-g[1])**2)
            if d < h:
                h = d
        return h

class Vfunction:
    def __init__(self, env) -> None:
        self.env = env
        self.values = {}

    def evaluate(self, state):
        if self.env.is_terminal(state):
            return self.env.get_terminal_cost(state)
        elif state in self.values.keys():
            return self.values[state]
        else:
            return self.env.heur(state)
        
    def get_Q_value(self, state, action, debug=False):
        succs = self.env.get_successors(state, action)
        vals = np.array([self.evaluate(s) for s in succs])
        ps = action.distribution
        if debug:
            print(f"State: {state}")
            print(f"Action: {action}")
            print(f"Successors: {succs}")
            print(f"Values: {vals}")
            print(f"Probs: {ps}")
        return np.sum((vals + action.costs) * ps)
    
    def get_best_action(self, state):
        min_h = np.inf
        best_action = None
        for a in self.env.get_applicable(state):
            h = self.get_Q_value(state, a)
            if h < min_h:
                min_h = h
                best_action = a
        return best_action, min_h
    
    def Bellman_update(self, state):
        _, val = self.get_best_action(state)
        self.values[state] = val
        return val

    # def display_best_actions(self):
    #     best_actions = []
    #     for y in range(self.env.map.shape[0]):
    #         row = []
    #         for x in range(self.env.map.shape[1]):
    #             row.append('  ')
    #         best_actions.append(row)
    #     for y in range(self.env.map.shape[0]):
    #         for x in range(self.env.map.shape[1]):
    #             s = State(x, y)
    #             if not self.env.is_terminal(s):
    #             # elif s in vfunc.values.keys():
    #                 best_actions[y][x], _ = self.get_best_action(s)

    #     for row in best_actions:
    #         for a in row:
    #             print(a, end=' | ')
    #         print()

    # def display_vfunc(self):
    #     im_val = np.zeros(self.env.map.shape)

    #     for y in range(im_val.shape[0]):
    #         for x in range(im_val.shape[1]):
    #             s = State(x, y)
    #             if self.env.is_terminal(s):
    #                 im_val[y, x] = 0
    #             # else:
    #             #     im_val[y, x] = self.evaluate(s) 
    #             if s in self.values.keys():
    #                 im_val[y,x] = self.values[s]
     
    #     plt.figure(2)
    #     plt.imshow(im_val, norm='linear')
    #     plt.colorbar()
    #     plt.show()

