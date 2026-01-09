from collections import deque
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import grid
import ghosts
import astar as search
import planning

def build_closed_subset(initial_state, env, vfunc):
    closed = deque()
    open = deque()
    open.append(initial_state)
    policy = {}

    while len(open) > 0:
        state = open.pop()
        closed.append(state)

        best_action, _ = vfunc.get_best_action(state)
        policy[state] = best_action
        # print(best_action)
        for s in env.get_successors(state, best_action):
            if s not in open and s not in closed and not env.is_terminal(s):
                open.append(s)
    return closed, policy

def display_closed_subset(env, closed):
    im_val = np.zeros(env.map.shape)

    for s in closed:
        im_val[s.y, s.x] = 20
    for y in range(im_val.shape[0]):
        for x in range(im_val.shape[1]):
            s = grid.State(x, y)
            if env.is_terminal(s):
                im_val[y, x] = 10
            # else:
            #     im_val[y, x]= self.evaluate(s) 
    
    plt.figure(3)
    plt.imshow(im_val, norm='linear')
    plt.show()


def ilao(initial_state, env, vfunc, epsilon):
    res = np.inf
    while res > epsilon:
        closed, policy = build_closed_subset(initial_state, env, vfunc)
        # print(closed)
        # vfunc.display_best_actions()
        # vfunc.display_vfunc()
        # display_closed_subset(env, closed)        

        stop = False
        while not stop and res > epsilon:        
            max_delta = 0
            for i in range(1):
                for state in closed:
                    # print(state)
                    v_old = vfunc.evaluate(state)
                    val = vfunc.Bellman_update(state)
                    # print(val)
                    a_new, v_new = vfunc.get_best_action(state)
                    # print(v_old, v_new)
                    if a_new != policy[state]:
                        stop = True
                    delta = abs(v_old - v_new)
                    if delta > max_delta:
                        # print(delta, max_delta)
                        max_delta = delta
            res = max_delta
            print(f"Residual: {res}")

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
            return heur(state)
        
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

    def is_safe(self, state, action, threshold=1000):
        succs = self.env.get_successors(state, action)
        for s in succs:
            if self.env.is_terminal(s) and not self.env.is_goal(s):
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
                s = grid.State(x, y)
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
                s = grid.State(x, y)
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

def heur(state):
    # return planning.expected_astar(state, env, vfunc_fixed, hfunc, penalty=False)
    # res = search.astar(state, env, vfunc_fixed, hfunc)
    # if res != None:
    #     return hfunc.evaluate(state)
    # else:
    return env.heur(state)


if __name__ == '__main__':

    np.random.seed(2)

    env = grid.Environment(7,7, grid.ACTIONS)
    env.generate_map(type=3, noise=False, prob=0.03)

    # env.display()

    vfunc = grid.Vfunction(env)
    vfunc_fixed = grid.Vfunction(env)
    hfunc = search.Hfunction(env, samples=1)

    state = env.start
    ilao(state, env, vfunc, 0.001)
    print(vfunc.get_best_action(state, debug=False))

    vfunc.display_best_actions()
    vfunc.display_vfunc()

    # s1 = grid.State(18, 9)
    # print(vfunc.get_best_action(s1))
    # print(f"s1 val = {vfunc.evaluate(s1)}")
    # vfunc.Bellman_update(s1)

    # s2 = grid.State(19, 9)
    # a2, v2 = vfunc.get_best_action(s2)
    # print(a2, v2)
    # vfunc.get_Q_value(s2, a2, debug=True)
    # print(vfunc.get_best_action(s2))
    # for a in env.get_applicable(s):
    #     val = vfunc.get_Q_value(s, a)
    #     print(f"Q({s}, {a}) = {val}")
