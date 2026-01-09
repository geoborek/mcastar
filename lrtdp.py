from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import grid
import astar as search
import planning

SAMPLED = True

def check_solved(state, solved, env, vfunc, eps):
    rv = True
    open = deque()
    closed = deque()

    if state not in solved:
        open.append(state)

    while len(open) > 0:
        s = open.pop()
        closed.append(s)

        # print(s)
        # print(residual(s, env, vfunc))
        if vfunc.residual(s) > eps:
            rv = False
            continue

        if not env.is_terminal(s):
            a, _ = vfunc.get_best_action(s)
            # print(a)
            succs = env.get_successors(s, a, sampled=SAMPLED)
            if SAMPLED:
                for t, _ in succs:
                    if t not in solved:
                        if t not in open and t not in closed:
                            open.append(t)
            else:
                for t in succs:
                    if t not in solved:
                        if t not in open and t not in closed:
                            open.append(t)
        
    if rv:
        for t in closed:
            solved.add(t)
    else:
        while len(closed) > 0:
            t = closed.pop()
            vfunc.Bellman_update(t, regularized=False)
    return rv

def lrtdp_trial(state, solved, env, vfunc, eps):    
    visited = deque()
    while not state in solved:
        visited.append(state)

        if env.is_terminal(state):
            break

        a, _ = vfunc.get_best_action(state)
        vfunc.Bellman_update(state, regularized=False)

        state,_ = env.get_sampled_successor(state, a)

    while len(visited) > 0:
        s = visited.pop()
        # vfunc.Bellman_update(s, regularized=False)
        if not check_solved(s, solved, env, vfunc, eps):
            break

def lrtdp(state, env, vfunc, eps=0.01, num_iter=100):
    solved = set()
    i = 0
    while state not in solved:
        if i % 10 == 0:
            print(f"Iteration: {i}, residual: {vfunc.residual(state)}")
            # print(solved)

        lrtdp_trial(state, solved, env, vfunc, eps)
        i += 1

class Vfunction:
    def __init__(self, env, vfunc_fixed, hfunc, hfunc_opp) -> None:
        self.env = env
        self.vfunc_fixed = vfunc_fixed
        self.hfunc = hfunc
        self.hfunc_opp = hfunc_opp
        self.env = env
        self.values = {}

    def evaluate(self, state, player=1):
        if self.env.is_terminal(state):
            return self.env.get_terminal_cost(state, player)
        elif state in self.values.keys():
            return self.values[state]
        else:
            if player==1:
                return heur(state, self.env, self.vfunc_fixed, self.hfunc, player=player)
            else:
                return heur(state, self.env, self.vfunc_fixed, self.hfunc_opp, player=player)
            
    def get_Q_value(self, state, action, sampled=False, debug=False, player=1):
        if sampled:
            succs = self.env.get_successors(state, action, sampled=True, num_samples=200)
            vals = np.array([c+self.evaluate(s, player=player) for s, c in succs])
            if debug:
                print(f"State: {state}")
                print(f"Action: {action}")
                print(f"Successors: {succs}")
                print(f"Values: {vals}")
            return np.mean(vals)
        else:
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
    
    def get_best_action(self, state, debug=False, regularized=False, beta=1, player=1):
        min_h = np.inf
        best_action = None
        actions = self.env.get_applicable(state)
        vals = np.zeros(len(actions))
        for i, a in enumerate(actions):
            h = self.get_Q_value(state, a, sampled=True, debug=False, player=player)
            vals[i] = h
            if debug:
                print(f"Action: {a}")
                print(f"Q-value: {h}")
            if h < min_h:
                min_h = h
                best_action = a
        if regularized:
            min_h = np.inner(vals, np.exp(vals/beta)/sum(np.exp(vals/beta)))

        return best_action, min_h

    def get_worst_action(self, state):
        max_h = -np.inf
        worst_action = None
        for a in self.env.get_applicable(state):
            h = self.get_Q_value(state, a, sampled=True)
            if h > max_h:
                max_h = h
                worst_action = a
        return worst_action, max_h

    def Bellman_update(self, state, regularized=False):
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
                    s = grid.State(x, y)
                    r = self.residual(s)
                    self.Bellman_update(s)
                    if r > res:
                        res = r

    def is_safe(self, state, action, threshold=1000):
        succs = self.env.get_successors(state, action)
        for s in succs:
            if self.env.is_terminal(s) and not self.env.is_goal(s):
                return False
            if self.evaluate(s) > threshold:
                return False
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

def heur(state, env, vfunc_fixed, hfunc, player=1):
    return planning.expected_astar(state, env, vfunc_fixed, hfunc, penalty=False, player=player)
    # res = search.astar(state, env, vfunc, hfunc)
    # if res != None:
    #     return hfunc.evaluate(state)
    # else:
    # return env.heur(state)

if __name__ == '__main__':

    np.random.seed(2)

    env = grid.Environment(7, 7, grid.ACTIONS)
    env.generate_map(type=3, noise=True, prob=0.03)

    # env.display()

    vfunc_fixed = grid.Vfunction(env)
    hfunc = search.Hfunction(env, samples=10)
    vfunc = Vfunction(env, vfunc_fixed, hfunc, hfunc)

    # val = planning.expected_astar(env.start, env, vfunc, hfunc, noise=0)
    # print(val)

    state = env.start #grid.State(0,2)
    lrtdp(state, env, vfunc, eps=0.001)
    print(vfunc.get_best_action(state, debug=True))
    # print(vfunc.get_Q_value(state, env.actions[3], debug=False, sampled=True))
    # vfunc.display_best_actions()
    # vfunc.display_vfunc()

    # vfunc2 = grid.Vfunction(env)
    # vfunc2.value_iteration(eps=0.01)
    # print(vfunc2.get_best_action(state, debug=False))
    # vfunc2.display_best_actions()
    # vfunc2.display_vfunc()



