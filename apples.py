import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod
import astar as search
import planning

COLOR_START = 20
COLOR_GOAL = 30
COLOR_WALL = 10
COLOR_PATH = 40
COLOR_AGENT = 50

COST_APPLE = 100

ACTIONS = [(' S', 1, (0, 1)),
           (' N', 1, (0,-1)),
        #    ('FS', 1, (0, 2)),
        #    ('FN', 1, (0,-2)),
           (' W', 1, (-1, 0)),
           (' E', 1, (1, 0)),
        #    ('SE', 1.414, (1, 1)),
        #    ('NE', 1.414, (1,-1)),
        #    ('SW', 1.414, (-1, 1)),
        #    ('NW', 1.414, (-1, -1))
           ] 

def shift(pos, vector):
    x,y = pos
    dx,dy = vector
    return (x+dx, y+dy)

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

def l1_distance(p1, p2):
    d = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    return d

@dataclass(eq=True, frozen=True)
class State:
    my_pos: tuple[int, int]
    agent_pos: tuple[int, int]
    apples: tuple[int,int] 

class Agent:
    def __init__(self, apples) -> None:
        self.apples = apples
        self.chosen_apple = None

    def get_closer(self, x,y):
        if x > y:
            return x - 1
        else:
            return x + 1

    def policy(self, pos, execute=False) -> tuple[int, int]:
        if execute:
            if pos == self.apples[0]:
                self.chosen_apple = self.apples[1]
            if pos == self.apples[1]:
                self.chosen_apple = self.apples[0]
        
        if self.chosen_apple == None:
            d0 = l1_distance(pos, self.apples[0])
            d1 = l1_distance(pos, self.apples[1])
            if d0 == d1:
                # If close to the center decision is up
                if pos[0]-self.apples[0][0] <= 0:
                    apple = self.apples[0]
                else:
                    i = np.random.randint(0,2)
                    apple = self.apples[i]
            elif d0 < d1:
                apple = self.apples[0]
            else:
                apple = self.apples[1]
        else:        
            apple = self.chosen_apple

        if pos[0] == apple[0]:
            x = pos[0]
            y = self.get_closer(pos[1], apple[1])
        elif pos[1] == apple[1]:
            y = pos[1]
            x = self.get_closer(pos[0], apple[0])
        else:
            d = np.random.randint(0,2)
            if d == 0:
                y = pos[1]
                x = self.get_closer(pos[0], apple[0])
            else:
                x = pos[0]
                y = self.get_closer(pos[1], apple[1])
        return (x,y)    

class Action:
    def __init__(self, name, cost, effect, agent):
        self.name = name
        self.cost = cost
        self.effect = effect
        self.agent = agent

    def __repr__(self):
        return self.name

    def sample(self, state, execute=False):
        apos = self.agent.policy(state.agent_pos, execute)
        mpos = shift(state.my_pos, self.effect)
        return State(mpos, apos, state.apples)

    def successors(self, state, num_samples=10):
        return [self.sample(state) for i in range(num_samples)]
    
class Environment:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.map = np.zeros((height, width))
        self.goals = [(self.width//2, 0), (self.width//2, self.height-1)]
        self.start = State((0, self.height//2), (self.width-1, self.height//2), (0,0))
        self.agent = Agent(self.goals)

        self.actions = [Action(name, cost, effect, self.agent) for (name, cost, effect) in ACTIONS]

    def generate_map(self):
        # self.map = COLOR_WALL*np.random.binomial(1,0.05,(self.height, self.width))
        # self.map[(2*self.height) // 3, 0:self.width-3] = COLOR_WALL
        # self.map[self.width // 3, 3:] = COLOR_WALL
        # self.map[2*self.height // 3, 0:-4] = COLOR_WALL
        # self.map[2*self.height // 3, -3:] = 0
        self.map[self.start.my_pos[1], self.start.my_pos[0]] = COLOR_START
        self.map[self.start.agent_pos[1], self.start.agent_pos[0]] = COLOR_AGENT
        for x,y in self.goals:
            self.map[y, x] = COLOR_GOAL

    def display(self):
        plt.imshow(self.map)
        plt.show()

    def display_path(self, path):
        m = np.copy(self.map)
        for i, state in enumerate(path):
            if self.is_inside(state):
                m[state.my_pos[1], state.my_pos[0]] = i*COLOR_PATH
        plt.figure(1)
        plt.imshow(m)

        m = np.copy(self.map)
        for i, state in enumerate(path):
            if self.is_inside(state):
                m[state.agent_pos[1], state.agent_pos[0]] = i*COLOR_PATH
        plt.figure(2)
        plt.imshow(m)
        plt.show()

    def is_inside(self, state):
        if state.my_pos[0] < 0 or state.my_pos[0] >= self.width:
            return False
        if state.my_pos[1] < 0 or state.my_pos[1] >= self.height:
            return False
        if state.agent_pos[0] < 0 or state.agent_pos[0] >= self.width:
            return False
        if state.agent_pos[1] < 0 or state.agent_pos[1] >= self.height:
            return False
        return True
    
    def project(self, state):
        # if self.is_inside(state):
        #     return state
        # else:
        x1 = state.my_pos[0]
        y1 = state.my_pos[1]
        x2 = state.agent_pos[0]
        y2 = state.agent_pos[1]
        return State((trunc(x1, self.width-1), trunc(y1, self.height-1)),
                     (trunc(x2, self.width-1), trunc(y2, self.height-1)),
                     state.apples)


    def is_goal(self, state):
        return state.apples == (1,1)

    def is_terminal(self, state):
        # print(state)
        n,m = state.apples
        return (abs(n)+abs(m)) == 2
        
    def get_terminal_cost(self, state):
        if self.is_goal(state):
            return 0
        else:
            return 0 #-state.apples[0]*COST_APPLE - state.apples[1]*COST_APPLE
        
    def get_applicable(self, state):
        if self.is_terminal(state):
            return []
        else:
            actions = self.actions
            return actions

    def get_successors(self, state, action, num_samples=20):
        return [self.get_sampled_successor(state, action) for i in range(num_samples)]
    
    def get_sampled_successor(self, state, action, execute=False):
        t = self.project(action.sample(state, execute))
        c = 0
        if t.my_pos == self.goals[0] and t.apples[0] == 0:
            new_state = State(t.my_pos, t.agent_pos, (1,t.apples[1]))
        elif t.agent_pos == self.goals[0] and t.apples[0] == 0:
            new_state = State(t.my_pos, t.agent_pos, (-1,t.apples[1]))
            c = COST_APPLE
        elif t.my_pos == self.goals[1] and t.apples[1] == 0:
            new_state = State(t.my_pos, t.agent_pos, (t.apples[0], 1))
        elif t.agent_pos == self.goals[1] and t.apples[1] == 0:
            new_state = State(t.my_pos, t.agent_pos, (t.apples[0], -1))
            c = COST_APPLE
        else:
            new_state = t

        return new_state, action.cost+c
    
    def heur(self, state):
        d = []        
        for g in self.goals:
            d.append(np.sqrt((state.my_pos[0]-g[0])**2 + (state.my_pos[1]-g[1])**2))
        if state.apples[0] != 0:
            h = d[1]
        elif state.apples[1] != 0:
            h = d[0]
        elif d[0] < d[1]:
            h = d[0] + self.height-1
        else:
            h = d[1] + self.height-1
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

    def is_safe(self, state, action, threshold=1000):
        # succs = self.env.get_successors(state, action)
        # for s in succs:
        #     print(s)
        #     if self.env.is_terminal(s) and not self.env.is_goal(s):
        #         return False
        #     if self.evaluate(s) > threshold:
        #         return False
        return True

    def get_Q_value(self, state, action):
        succs = self.env.get_successors(state, action)
        vals = np.array([c+self.evaluate(s) for s, c in succs])
        return np.mean(vals)
    
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

if __name__ == '__main__':

    # np.random.seed(3)
    # random.seed(2)

    env = Environment(9,9)
    env.generate_map()
    # env.display()

    # print(env.get_sampled_successor(State((2,1),(5,2),(0,0)), env.actions[5]))

    vfunc = Vfunction(env)
    hfunc = search.Hfunction(env, samples=50)

    planning.planning(env, vfunc, hfunc)
