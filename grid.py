import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
import astar as search

COLOR_START = 20
COLOR_GOAL = 30
COLOR_WALL = 10
COLOR_PATH = 40

COST_WALL = 1000

DIST = [0.8, 0.1, 0.1]
COSTS1 = [1, 1.414, 1.414]
COSTS2 = [1.414, 1, 1]
ACTIONS2 = [
           (' S', [1, 2, 3], [(0, 1), (0, 2), (0, 3)], [0.5, 0.4, 0.1]),
           (' N', [1, 2, 3], [(0,-1), (0,-2), (0, -3)], [0.5, 0.4, 0.1]),
           (' W', [1, 2, 3], [(-1, 0), (-2, 0), (-3, 0)], [0.5, 0.4, 0.1]),
           (' E', [1, 2, 3], [(1, 0), (2, 0), (3, 0)], [0.5, 0.4, 0.1]),
        #    (' SE', [1.5, 3], [(1, 1), (2, 2)], [0.6, 0.4]),
        #    (' NE', [1.5, 3], [(1,-1), (2,-2)], [0.6, 0.4]),
        #    (' SW', [1.5, 3], [(-1, 1), (-2, 2)], [0.6, 0.4]),
        #    (' NW', [1.5, 3], [(-1, -1), (-2, -2)], [0.6, 0.4]),
           ]

ACTIONS = [(' S', COSTS1, [(0, 1), (1, 1), (-1, 1)], DIST),
           (' N', COSTS1, [(0,-1), (1,-1), (-1,-1)], DIST),
           (' W', COSTS1, [(-1, 0), (-1, 1), (-1, -1)], DIST),
           (' E', COSTS1, [(1, 0), (1, 1), (1, -1)], DIST),
           ('SE', COSTS2, [(1, 1), (0, 1), (1, 0)], DIST),
           ('NE', COSTS2, [(1,-1), (1,0), (0,-1)], DIST),
           ('SW', COSTS2, [(-1, 1), (-1, 0), (0, 1)], DIST),
           ('NW', COSTS2, [(-1, -1), (-1, 0), (0, -1)], DIST),
           ]

@dataclass(eq=True, frozen=True)
class State:
   x: int
   y: int

@dataclass
class Vector:
   x: int
   y: int

def shift(state, vector):
    return State(state.x+vector.x, state.y+vector.y)

class Action(ABC):
    def __init__(self, name, costs, distribution):
        self.name = name
        self.costs = np.array(costs)
        self.distribution = np.array(distribution)

    def __repr__(self):
        return self.name
    
    # def __eq__(self, action) -> bool:
    #     return self.name == action.name
    
    @abstractmethod
    def sample(self, state) -> tuple[State, np.number]:
        pass

    @abstractmethod
    def successors(self, state) -> list[State]:
        pass

class ShiftAction(Action):
    def __init__(self, name, costs, distribution, effects):
        super().__init__(name, costs, distribution)        
        self.effects = [Vector(eff[0], eff[1]) for eff in effects]

    def successors(self, state):
        return [shift(state, eff) for eff in self.effects]
    
    def sample(self, state):
        samples = np.random.choice(range(len(self.effects)), size=1, p=self.distribution)
        return shift(state, self.effects[samples[0]]), self.costs[samples[0]]

class Environment:
    def __init__(self, width, height, actions) -> None:
        self.width = width
        self.height = height
        self.map = np.zeros((height, width))
        self.actions = [ShiftAction(name, costs, dist, effects) for name, costs, effects, dist in actions]
        self.start = State(0, self.height-1)
        self.goals = [State(self.width-1, 1), # self.height//2 - 1), # 0),
                      State(self.width-1, 2), #self.height//2), # 1),
                      State(self.width-1, 3), #self.height//2 + 1), #2),
                    #   State(self.width-2, 1),
                    #   State(self.width-2, 2),
                    #   State(self.width-2, 3),
                    #   State(self.width-3, 1),
                    #   State(self.width-3, 2),
                    #   State(self.width-3, 3),
                      ]
        self.successors = {}

    def generate_map(self, type=0, noise=False, prob=0.01):
        if noise:
            self.map = COLOR_WALL*np.random.binomial(1,prob,(self.height, self.width))
        
        if type == 0:
            # self.map[self.width // 2+2, 0:(2*self.width//3)] = COLOR_WALL
            self.map[self.width // 2-2, 3:] = COLOR_WALL
            self.map[(self.width // 2)-1:(self.width // 2)+1, :] = 0
        elif type == 1:
            self.map[self.width // 2+1, 0:(2*self.width//3)] = COLOR_WALL
            self.map[self.width // 2-2, 3:] = COLOR_WALL
            self.map[self.width // 2-1, :] = 0
            self.map[self.width // 2, :] = 0
            self.goals = [State(self.width-1, 1), 
                          State(self.width-1, 2), 
                          State(self.width-1, 3), 
                          State(self.width-2, 1),
                          State(self.width-2, 2),
                          State(self.width-2, 3),
                          State(self.width-3, 1),
                          State(self.width-3, 2),
                          State(self.width-3, 3),
                        ]
        elif type==3:
            self.start = State(0, self.height-1)
            self.map[self.width // 2, 0:(2*self.width//3)] = COLOR_WALL
            self.goals = [State(self.width-1, 0), 
                        State(self.width-1, 1),
                        State(self.width-1, 2)]
            # self.map[self.height-1, self.width-1] = COLOR_WALL

        self.map[self.start.y, self.start.x] = COLOR_START
        for state in self.goals:
            self.map[state.y, state.x] = COLOR_GOAL

    def display(self):
        plt.imshow(self.map)
        plt.show()

    def display_path(self, path):
        m = np.copy(self.map)
        for state in path:
            if self.is_inside(state):
                m[state.y, state.x] = COLOR_PATH
        plt.figure(1)
        plt.imshow(m)
        plt.show()

    def is_inside(self, state):
        return state.x >= 0 and state.x < self.width and state.y >=0 and state.y < self.height
    
    def project(self, state, safe=True):
        if not safe:
            return state
        if self.is_inside(state):
            return state
        else:
            x = state.x
            y = state.y    
            if x < 0:
                x = 0
            elif x >= self.width:
                x = self.width - 1
            if y < 0:
                y = 0
            elif y >= self.height:
                y = self.height - 1
            return State(x, y)

    def is_wall(self, state):
        return self.map[state.y, state.x] == COLOR_WALL

    def is_goal(self, state):
        return state in self.goals

    def is_terminal(self, state):
        return self.is_goal(state) or not self.is_inside(state) or self.is_wall(state)
        
    def get_terminal_cost(self, state, player=1):
        if self.is_goal(state):
            return 0
        else:
            return COST_WALL
        
    def get_applicable(self, state):
        if self.is_terminal(state):
            return []
        else:
            actions = self.actions
            return actions

    # def get_successors(self, state, action):
    #     return [self.project(s) for s in action.successors(state)]
    
    def get_sampled_successor(self, state, action, execute=False, safe=True):
        t, cost = action.sample(state)
        if safe and self.is_wall(self.project(t)):
            return state, cost
        return self.project(t), cost

    def get_successors(self, state, action, sampled=False, num_samples=20, cache=True):
        if sampled:
            if cache and (state, action) in self.successors:
                return self.successors[(state, action)]
            elif cache:
                self.successors[(state, action)] = [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
                return self.successors[(state, action)]
            else:
                return [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
        else:
            return [self.project(s) for s in action.successors(state)]

    def get_extrem_successor(self, state, action, worst=True, beta=0.1):
        succs = self.get_successors(state, action)
        vals = action.costs + np.array([self.heur(t) for t in succs]) 
        if not worst:
            vals = -vals 
        vals += np.random.gumbel(0, beta, len(succs))
        i = np.argmax(vals)
        return succs[i], action.costs[i]
            
    def heur(self, state):
        h = np.inf
        for g in self.goals:
            d = np.sqrt((state.x-g.x)**2 + (state.y-g.y)**2)
            # d = (abs(state.x-g.x) + abs(state.y-g.y))/3
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
            print(f"Costs: {action.costs}")
            print(f"Probs: {ps}")
        return np.sum((vals + action.costs) * ps)
    
    def get_best_action(self, state, debug=False):
        min_h = np.inf
        best_action = None
        for a in self.env.get_applicable(state):
            h = self.get_Q_value(state, a)
            if debug:
                print(f"Action: {a}")
                print(f"Q-value: {h}")
            if h < min_h:
                min_h = h
                best_action = a
        return best_action, min_h
    
    def Bellman_update(self, state):
        _, val = self.get_best_action(state)
        self.values[state] = val
        return val

    def residual(self, state):
        if self.env.is_terminal(state):
            return 0
        else:
            val = self.evaluate(state) 
            _, new_val = self.get_best_action(state)
            return abs(new_val-val) 

    def value_iteration(self, eps=0.01):
        res = np.inf
        while res > eps:
            res = 0
            for y in range(self.env.map.shape[0]):
                for x in range(self.env.map.shape[1]):
                    s = State(x, y)
                    r = self.residual(s)
                    self.Bellman_update(s)
                    if r > res:
                        res = r

    def is_safe(self, state, action, threshold=1000):
        succs = self.env.get_successors(state, action)
        for s in succs:
            if self.env.is_terminal(s) and not self.env.is_goal(s):
                # print(s)
                return False
            # if self.evaluate(s) > threshold:
            #     return False
        return True

    def display_best_actions(self):
        best_actions = []
        for y in range(self.env.map.shape[0]):
            row = []
            for x in range(self.env.map.shape[1]):
                row.append('  ')
            best_actions.append(row)
        for y in range(self.env.map.shape[0]):
            for x in range(self.env.map.shape[1]):
                s = State(x, y)
                if not self.env.is_terminal(s):
                # elif s in vfunc.values.keys():
                    best_actions[y][x], _ = self.get_best_action(s)

        for row in best_actions:
            for a in row:
                print(a, end=' | ')
            print()

    def display_vfunc(self):
        im_val = np.zeros(self.env.map.shape)

        for y in range(im_val.shape[0]):
            for x in range(im_val.shape[1]):
                s = State(x, y)
                if self.env.is_terminal(s):
                    im_val[y, x] = 0
                # else:
                #     im_val[y, x] = self.evaluate(s) 
                if s in self.values.keys():
                    im_val[y,x] = self.values[s]
     
        plt.figure(2)
        plt.imshow(im_val, norm='linear')
        plt.colorbar()
        plt.show()

