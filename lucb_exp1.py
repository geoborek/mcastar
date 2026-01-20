from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import grid
import astar as search
import planning
import test_env as test
import pandas as pd
import time

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

def lrtdp_trial(state, solved, env, vfunc, eps, check=True):    
    visited = deque()
    while not state in solved:
        visited.append(state)

        if env.is_terminal(state) or len(visited)>100:
            break

        a, _ = vfunc.get_best_action(state)
        vfunc.Bellman_update(state, regularized=False)

        state,_ = env.get_sampled_successor(state, a)

    if check:
        while len(visited) > 0:
            s = visited.pop()
            # vfunc.Bellman_update(s, regularized=False)
            if not check_solved(s, solved, env, vfunc, eps):
                break

def lrtdp(state, env, vfunc, eps=0.01, iterations=100, check=True, log=False):
    solved = set()
    actions = env.get_applicable(state)

    if log:
        pvals = np.zeros((iterations, len(actions)))
        tvals = np.zeros(iterations)

    i = 0
    start = time.time()
    while state not in solved and i<iterations:
    # for k in range(10):
        if i % 1000 == 0:
            print(f"Iteration: {i}, residual: {vfunc.residual(state)}")
            # print(vfunc.get_Q_value(env.start, env.actions[0]))
            # print(solved)

        lrtdp_trial(state, solved, env, vfunc, eps, check=check)
        if log:
            for j, a in enumerate(actions):
                pvals[i,j] = vfunc.get_Q_value(state, a, sampled=True, num_samples=100)
            tvals[i] = time.time()-start

        i += 1

    if log:
        df = pd.DataFrame(pvals, columns=actions)
        dft = pd.DataFrame(tvals, columns=['time'])
        return df, dft


def get_two_best(vals):
    arr = np.copy(vals)
    fst = np.argmin(arr)
    arr[fst] = np.inf
    snd = np.argmin(arr)
    return fst, snd

class Vfunction:
    def __init__(self, env, vfunc_fixed, hfunc, hfunc_opp, heur_type="base", confidence=0.1, eps=0.1) -> None:
        self.env = env
        self.vfunc_fixed = vfunc_fixed
        self.hfunc = hfunc
        self.hfunc_opp = hfunc_opp
        self.heur_type = heur_type
        self.confidence = confidence
        self.eps = eps
        self.env = env
        self.values = {}
        self.visited = {}
        self.outcomes = {}

    def evaluate(self, state, player=1):
        if self.env.is_terminal(state):
            return self.env.get_terminal_cost(state, player)
        elif state in self.values.keys():
            return self.values[state]
        else:
            if player==1:
                return heur(state, self.env, self.vfunc_fixed, self.hfunc, type=self.heur_type, player=player)
            else:
                return heur(state, self.env, self.vfunc_fixed, self.hfunc_opp, type=self.heur_type, player=player)
       
    def get_Q_value(self, state, action, sampled=False, debug=False, num_samples=1, player=1):
        if sampled:
            succs = self.env.get_successors(state, action, sampled=True, num_samples=num_samples)
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

    def get_confidence_bound(self, vals, n, k, threshold=1000):
        m = np.max(vals)
        if m >= threshold:
            return np.inf
        
        delta = np.max(vals)- np.min(vals)
        return delta*np.sqrt(0.5*np.log(k/self.confidence)/n)

    def update_outcome(self, state, action):
        succs = self.env.get_successors(state, action, sampled=False)
        t, _ = self.env.get_sampled_successor(state, action)
        if (state, action) not in self.visited:
            self.visited[(state, action)] = 1
            distribution = np.zeros(len(succs))
            distribution[succs.index(t)] = 1
            self.outcomes[(state, action)] = distribution
        else:
            self.visited[(state, action)] += 1
            self.outcomes[(state, action)][succs.index(t)] += 1

    def LUCB(self, state, threshold=1000):
        actions = self.env.get_applicable(state)
        up_vals = np.zeros(len(actions))
        low_vals = np.zeros(len(actions))
        confidence = np.zeros(len(actions))
        target_vals = {}
        succs = {}

        # try each action once
        for i, a in enumerate(actions):
            succs[a] = self.env.get_successors(state, a, sampled=False)
            target_vals[a] = np.array([c+self.evaluate(t) for t, c in zip(succs[a], a.costs)])
            self.update_outcome(state, a)
            confidence[i] = self.get_confidence_bound(target_vals[a], self.visited[(state, a)], 4, threshold=threshold)
            expected_val = np.inner(self.outcomes[(state, a)], target_vals[a])/np.sum(self.outcomes[(state, a)])
            if confidence[i] < np.inf:
                low_vals[i] = expected_val - confidence[i]        
                up_vals[i] = expected_val + confidence[i]        
            else: 
                low_vals[i] = threshold       
                up_vals[i] = threshold

        # print(state, low_vals)
        fst, snd = get_two_best(low_vals)
        k = 0
        while up_vals[fst] > low_vals[snd] and (up_vals[fst]-low_vals[fst]) > self.eps:
            k += 1
            self.update_outcome(state, actions[fst])
            self.update_outcome(state, actions[snd])

            for i, a in enumerate(actions):
                confidence[i] = self.get_confidence_bound(target_vals[a], self.visited[(state, a)], 4, threshold=threshold)
                expected_val = np.inner(self.outcomes[(state, a)], target_vals[a])/np.sum(self.outcomes[(state, a)])
                if confidence[i] < np.inf:
                    low_vals[i] = expected_val - confidence[i]        
                    up_vals[i] = expected_val + confidence[i]        
                else: 
                    low_vals[i] = threshold      
                    up_vals[i] = threshold

            fst, snd = get_two_best(low_vals)

            if k % 1000 == 0:
                # print(up_vals, low_vals)
                print(state, k, actions[fst], actions[snd], (up_vals-low_vals)[fst])

        # if up_vals[fst] > low_vals[snd]:
        #     print("LUCB proof!")
        best_action = actions[fst]
        while confidence[fst] > self.eps and confidence[fst] < np.inf:
            k += 1
            self.update_outcome(state, actions[fst])
            confidence[fst] = self.get_confidence_bound(target_vals[best_action], self.visited[(state, best_action)], 2)
            # expected_val = np.inner(self.outcomes[(state, best_action)], target_vals[best_action])/np.sum(self.outcomes[(state, best_action)])
            # low_vals[fst] = expected_val - confidence[fst]        
            # up_vals[fst] = expected_val + confidence[fst]        

            if k % 1000 == 0:
                # print(up_vals, low_vals)
                print(state, k, actions[fst], confidence[fst], (up_vals-low_vals)[fst])

        expected_val = np.inner(self.outcomes[(state, best_action)], target_vals[best_action])/np.sum(self.outcomes[(state, best_action)])

        return actions[fst], expected_val

    def get_best_action(self, state, debug=False, regularized=False, beta=1, player=1):
        min_h = np.inf
        best_action = None
        actions = self.env.get_applicable(state)
        vals = np.zeros(len(actions))

        for i, a in enumerate(actions):
            h = self.get_Q_value(state, a, sampled=True, debug=False, num_samples=100, player=player)
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
        # _, val = self.get_best_action(state)
        if self.env.is_terminal(state):
            val = self.env.get_terminal_cost(state)
            self.values[state] = val
        else:
            _, val = self.LUCB(state)
            self.values[state] = val
        return val
    
    def residual(self, state):
        if self.env.is_terminal(state):
            return 0
        else:
            val = self.evaluate(state) 
            _, new_val = self.LUCB(state) #self.get_best_action(state)
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
                    best_actions[y][x], _ = self.LUCB(s) #self.get_best_action(s)

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
        plt.axis('off')
        # plt.colorbar()
        plt.show()

def heur(state, env, vfunc_fixed, hfunc, type="expected", player=1):
    if type == "blind":
        return 0
    elif type == "base":
        return env.heur(state)
    else:
        return planning.expected_astar(state, env, vfunc_fixed, hfunc, penalty=False, player=player)
    

if __name__ == '__main__':

    plt.rcParams.update({'font.size': 24})
    plt.rcParams['text.usetex'] = True

    np.random.seed(3)

    env = grid.Environment(20, 20, grid.ACTIONS, safe=False)
    env.generate_map(type=4, noise=False, prob=0.02)
    env.display()

    # env = test.env

    vfunc_fixed = grid.Vfunction(env)
    hfunc = search.Hfunction(env, samples=10)

    # val = planning.expected_astar(env.start, env, vfunc, hfunc, noise=0)
    # print(val)

    state = env.start #grid.State(0,2)

    ITERATIONS = 200
    vfunc = Vfunction(env, vfunc_fixed, hfunc, hfunc, heur_type="blind", confidence=0.05, eps=0.5)
    df_bl, dft_bl = lrtdp(state, env, vfunc, eps=0.01, iterations=ITERATIONS, check=False, log=True)
    # print(vfunc.LUCB(state))
    print(vfunc.get_best_action(state, debug=True))

    # print(vfunc.get_Q_value(state, env.actions[3], debug=True, sampled=True, num_samples=100))
    # vfunc.display_best_actions()
    vfunc.display_vfunc()

    vfunc = Vfunction(env, vfunc_fixed, hfunc, hfunc, heur_type="base", confidence=0.05, eps=0.5)
    df_bs, dft_bs = lrtdp(state, env, vfunc, eps=0.01, iterations=ITERATIONS, check=False, log=True)
    # print(vfunc.LUCB(state))
    print(vfunc.get_best_action(state, debug=True))
    vfunc.display_vfunc()

    vfunc = Vfunction(env, vfunc_fixed, hfunc, hfunc, heur_type="expected", confidence=0.05, eps=0.5)
    df_ex, dft_ex = lrtdp(state, env, vfunc, eps=0.01, iterations=ITERATIONS, check=False, log=True)
    # print(vfunc.LUCB(state))
    print(vfunc.get_best_action(state, debug=True))
    vfunc.display_vfunc()

    plt.figure(1)
    plt.plot(range(1,ITERATIONS+1), 53*np.ones(ITERATIONS), label=r'$Q^*(s_0,W)$')
    actions = env.get_applicable(state)
    for i, a in enumerate(actions):
        if a.name in [' W']:
            plt.plot(range(1,ITERATIONS+1), df_bl[a], label='blind')

    for i, a in enumerate(actions):
        if a.name in [' W']:
            plt.plot(range(1,ITERATIONS+1), df_bs[a], label='euclid')

    for i, a in enumerate(actions):
        if a.name in [' W']:
            plt.plot(range(1,ITERATIONS+1), df_ex[a], label='RA*')

    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\hat{Q}$-values')
    plt.legend(loc="lower right")    
    plt.show()

    plt.figure(2)
    plt.plot(range(1,ITERATIONS+1), dft_bl, label='blind')
    plt.plot(range(1,ITERATIONS+1), dft_bs, label='euclid')
    plt.plot(range(1,ITERATIONS+1), dft_ex, label='RA*')
    plt.xlabel('Iteration')
    plt.ylabel(r'time [s]')
    plt.legend(loc="upper left")    

    plt.show()



