import numpy as np
from collections import deque
import grid
import astar as search
from tqdm import tqdm
import test_env as test
import lrtdp
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import planning
import lucb_lrtdp as lucb
import time

def rollout_action(state, qfunc, state_visited, mode="exploration", exploration=1.41):
    actions = env.get_applicable(state)
    if mode == "uniform":
        return np.random.choice(actions)

    if mode == "noise":
        best_action = None
        min_h = np.inf
        for a in actions:
            h = get_heur_Q_value(state, a, env, vfunc_fixed, hfunc) - np.random.gumbel(0, 0.5)
            if h < min_h:
                min_h = h
                best_action = a
        return best_action

    if mode == "exploration":
        untried_actions = []
        for a in actions:
            if (state, a) not in qfunc.pair_visited:
                untried_actions.append(a)

        # if len(untried_actions)>0:
        #     return np.random.choice(untried_actions)        
        best_action = None
        min_h = np.inf
        for a in untried_actions:
            h = get_heur_Q_value(state, a, env, vfunc_fixed, hfunc) - np.random.gumbel(0, 1)
            if h < min_h:
                min_h = h
                best_action = a
        if best_action != None:
            return best_action
    a, _ = qfunc.get_best_action(state, state_visited, mode=mode, exploration=exploration)
    return a

def rollout(state, env, qfunc, state_visited, mode="exploration", max_iter=1000, exploration=1.41):
    open = deque()

    s = state
    total_cost = 0
    k = 0 
    while not env.is_terminal(s) and total_cost<1000: #k<max_iter:
        a = rollout_action(s, qfunc, state_visited, mode=mode, exploration=exploration)
        if a == None:
            break
        if mode == "best":
            t, cost = qfunc.histograms[(s, a)].sample()
        else:
            t, cost = env.get_sampled_successor(s, a)
        open.append((s, a, cost, t))
        s = t 
        total_cost += cost
        k += 1
    
    if env.is_terminal(s):
        terminal_cost = env.get_terminal_cost(s)
    else:
        terminal_cost = heur(s, env, vfunc_fixed, hfunc)
    # if mode == "best":
    # print(f"terminal: {terminal_cost}, total: {total_cost}, iterations: {k}")
    # if total_cost >= 0:
    #     print(open)
    return open, terminal_cost, total_cost+terminal_cost, k

def update(trace, qfunc, state_visited, terminal_cost, init=0, discount=True):
    cost = terminal_cost
    while len(trace) > 0:
        s, a, c, t = trace.pop()
        if s in state_visited:
            state_visited[s] += 1
        else:
            state_visited[s] = init+1
        cost += c
        qfunc.update(s, a, cost, t, c, init=init, discount=discount)
    return 0

def mcts(state, env, qfunc, iterations=80000, exploration=7, init=20, discount=True, log=False):
    state_visited = {}
    actions = env.get_applicable(state)

    if log:
        pvals = np.zeros((iterations, len(actions)))
        tvals = np.zeros(iterations)
        iter_vals = np.zeros(iterations)
        cost_vals = np.zeros(iterations)

    start = time.time()
    for i in tqdm(range(iterations)):
        trace, terminal_cost, total_cost, iter = rollout(state, env, qfunc, state_visited, 
                                                         mode="exploration", exploration=exploration)
        update(trace, qfunc, state_visited, terminal_cost, init=init, discount=discount)

        if log:
            for j, a in enumerate(actions):
                pvals[i,j] = qfunc.evaluate(state, a)
            tvals[i] = time.time()-start
            iter_vals[i] = iter
            cost_vals[i] = total_cost

    best_action, val = qfunc.get_best_action(state,state_visited, mode="best")

    plt.figure(2)
    plt.plot(range(1,iterations+1), iter_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Rollout size')
    plt.xscale('log')
    # plt.yscale('log')
    plt.show()
    plt.figure(2)
    plt.plot(range(1,iterations+1), cost_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Rollout cost')
    plt.xscale('log')
    # plt.yscale('log')

    plt.show()
    if log:
        df = pd.DataFrame(pvals, columns=actions)
        dft = pd.DataFrame(tvals, columns=['time'])
        return df, dft


def estimate(state, env, qfunc, state_visited):
    _, terminal_cost, total_cost, _ = rollout(state, env, qfunc, state_visited, mode="best")
    # print(terminal_cost, total_cost)

    return total_cost #min(total_cost, 50)

def brue_update(trace, env, qfunc, state_visited):
    while len(trace) > 0:
        s, a, c, t = trace.pop()
        if s in state_visited:
            state_visited[s] += 1
        else:
            state_visited[s] = 1
        cost = estimate(t, env, qfunc, state_visited)
        cost += c
        qfunc.update(s, a, cost, t, c)
    return 0

def brue(state, env, qfunc, iterations=2000, log=False):
    state_visited = {}
    actions = env.get_applicable(state)

    if log:
        pvals = np.zeros((iterations, len(actions)))
        tvals = np.zeros(iterations)

    start = time.time()
    for i in tqdm(range(iterations)):
        trace, terminal_cost, total_cost, _ = rollout(state, env, qfunc, state_visited, mode="uniform")
        brue_update(trace, env, qfunc, state_visited)

        if log:
            for j, a in enumerate(actions):
                pvals[i,j] = qfunc.evaluate(state, a)
            tvals[i] = time.time() - start

    best_action, val = qfunc.get_best_action(state, state_visited, mode="best")

    if log:
        df = pd.DataFrame(pvals, columns=actions)
        dft = pd.DataFrame(tvals, columns=['time'])
        return df, dft
    return best_action, val

class Histogram:
    def __init__(self, state, cost):
        self.targets = [state]
        self.costs = [cost]
        self.visits = [1]

    def update(self, state, cost):
        if state in self.targets:
            self.visits[self.targets.index(state)] += 1
        else:
            self.targets.append(state)
            self.costs.append(cost)
            self.visits.append(1)

    def sample(self):
        t = np.random.choice(self.targets, p=self.visits/np.sum(self.visits))
        return t, self.costs[self.targets.index(t)]
    
class Qfunction:
    def __init__(self, env, vfunc, hfunc) -> None:
        self.env = env
        self.vfunc = vfunc
        self.hfunc = hfunc
        self.values = {}
        self.pair_visited = {}
        self.histograms = {}

    def evaluate(self, state, action):
        if (state, action) in self.values.keys():
            return self.values[(state, action)]
        else:
            return get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc)

    def weight(self, n):
        return n
    
    def weight_sum(self, n):
        return n*(n+1)/2

    def update(self, state, action, cost, target, acost, init=20, discount=False):
        if (state, action) in self.pair_visited:
            n = self.pair_visited[(state, action)]
            # if state==self.env.start and action==self.env.actions[3]:
            #     print(f"Value: {self.values[(state, action)]}")
            if discount:
                self.values[(state, action)] = (self.values[(state, action)]*self.weight_sum(n) + self.weight(n+1)*cost)/self.weight_sum(n+1)
            else:
                self.values[(state, action)] = (self.values[(state, action)]*n + cost)/(n+1)
            # if state==self.env.start and action==self.env.actions[3]:
            #     print(f"Updated Value: {self.values[(state, action)]}")
            self.pair_visited[(state, action)] += 1
            self.histograms[(state, action)].update(target, acost)
            # if state==self.env.start and action==self.env.actions[5]:
            #     print(f"Update value: {self.values[(state, action)]}")
            #     print(get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc), cost, init)
        else:            
            h = get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc)
            self.values[(state, action)] = (h*init + cost)/(init+1)
            self.pair_visited[(state, action)] = init+1
            self.histograms[(state, action)] = Histogram(target, acost)
            # if state==self.env.start and action==self.env.actions[5]:
            #     print(f"First value: {self.values[(state, action)]}")
            #     print(get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc), cost, init)

    def get_best_action(self, state, state_visited, mode="exploration", debug=False, exploration=1.41):
        min_h = np.inf
        best_action = None
        for a in self.env.get_applicable(state):
            if (state, a) in self.values:
                h = self.evaluate(state, a) 
                if mode == "exploration":
                    # print(exploration)
                    h -= exploration*np.sqrt(np.log(state_visited[state])/self.pair_visited[(state, a)])
                if debug:
                    print(f"Action: {a}")
                    print(f"Q-value: {h}")
                if h < min_h:
                    min_h = h
                    best_action = a

        return best_action, min_h

    def display_qfunc(self):
        im_val = np.zeros(self.env.map.shape)

        for y in range(im_val.shape[0]):
            for x in range(im_val.shape[1]):
                s = grid.State(x, y)
                if self.env.is_terminal(s):
                    im_val[y, x] = 0
                # else:
                #     im_val[y, x] = self.evaluate(s) 
                else:
                    a, v = self.get_best_action(s, {}, mode="best")
                    if v != None:
                        im_val[y,x] = v
     
        plt.figure(2)
        plt.imshow(im_val, norm='linear')
        plt.colorbar()
        plt.show()


def get_heur_Q_value(state, action, env, vfunc, hfunc):
    # succs = env.get_successors(state, action, sampled=True, num_samples=50)
    # res = 0
    # for t, c in succs:
    #     h = heur(t, env, vfunc, hfunc, player=1)
    #     res += c + h
    # return res/len(succs)
    t, c = env.get_extrem_successor(state, action, worst=False, beta=0)
    return heur(t, env, vfunc, hfunc)+c

def heur(state, env, vfunc, hfunc, player=1):
    return planning.expected_astar(state, env, vfunc, hfunc, penalty=False, player=player)
    # res = search.astar(state, env, vfunc, hfunc)
    # if res != None:
    #     return hfunc.evaluate(state)
    # else:
    # return env.heur(state)

if __name__ == '__main__':

    np.random.seed(3)
    plt.rcParams.update({'font.size': 24})
    plt.rcParams['text.usetex'] = True

    env = grid.Environment(10, 10, grid.ACTIONS, safe=True)
    env.generate_map(type=0, noise=True, prob=0.05)
    env.display()

    # env = test.env

    vfunc_fixed = grid.Vfunction(env)
    hfunc = search.Hfunction(env, samples=10)
    vfunc = lrtdp.Vfunction(env, vfunc_fixed, hfunc, hfunc)

    qfunc = Qfunction(env, vfunc_fixed, hfunc)

    state = env.start
    # for a in env.actions:
    #     print(env.get_extrem_successor(state, a))
    #     print(get_heur_Q_value(state, a, env, vfunc, hfunc))

    # lrtdp.lrtdp(state, env, vfunc, eps=0.01)
    # print(vfunc.get_best_action(state, debug=False))
    # print(vfunc.get_Q_value(state, env.actions[5]))

    val = planning.expected_astar(state, env, vfunc, hfunc, noise=0)
    print(val)

    ITERATIONS = 4000
    df_uct, dft_uct = mcts(state, env, qfunc, iterations=ITERATIONS, discount=False, exploration=50, log=True)
    # print(a, val)

    # qfunc = Qfunction(env, vfunc_fixed, hfunc)
    # df = mcts(state, env, qfunc, iterations=ITERATIONS, discount=True, exploration=2000, log=True)

    qfunc = Qfunction(env, vfunc_fixed, hfunc)
    df_brue, dft_brue = brue(state, env, qfunc, iterations=ITERATIONS, log=True)
    # print(df)

    vfunc = lucb.Vfunction(env, vfunc_fixed, hfunc, hfunc, heur_type="expected", confidence=0.1, eps=0.1)
    df_lucb, dft_lucb = lucb.lrtdp(state, env, vfunc, eps=0.1, iterations=ITERATIONS, check=False, log=True)

    # for a in env.get_applicable(state):
    #     if (state, a) in qfunc.values:
    #         print(a, qfunc.values[(state, a)])
    #         # print(get_heur_Q_value(state, a, env, vfunc_fixed, hfunc))
    # # qfunc.display_qfunc()

    plt.figure(1)
    plt.plot(range(1,ITERATIONS+1), 14.33*np.ones(ITERATIONS), label=r'$Q^*(s_0,NE)$')
    actions = env.get_applicable(state)
    for i, a in enumerate(actions):
        if a.name in [' N', 'NE', 'NW']:
            plt.plot(range(1,ITERATIONS+1), df_uct[a], label=a)

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\hat{Q}$-values')
    plt.legend(loc="upper left")    
    plt.title('UCT')
    plt.show()

    plt.plot(range(1,ITERATIONS+1), 14.33*np.ones(ITERATIONS), label=r'$Q^*(s_0,NE)$')
    actions = env.get_applicable(state)

    for i, a in enumerate(actions):
        if a.name in [' N', 'NE', 'NW']:
            plt.plot(range(1,ITERATIONS+1), df_brue[a], label=a)

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\hat{Q}$-values')
    plt.legend(loc="upper right")  
    plt.title('BRUE')  
    plt.show()

    plt.plot(range(1,ITERATIONS+1), 14.33*np.ones(ITERATIONS), label=r'$Q^*(s_0,NE)$')
    actions = env.get_applicable(state)
    for i, a in enumerate(actions):
        if a.name in [' N', 'NE', 'NW']:
            plt.plot(range(1,ITERATIONS+1), df_lucb[a], label=a)

    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\hat{Q}$-values')
    plt.legend(loc="lower right")  
    plt.title('LUCB-RTDP')  
    plt.show()

    plt.plot(range(1,ITERATIONS+1), dft_uct, label='UCT')
    plt.plot(range(1,ITERATIONS+1), dft_brue, label='BRUE')
    plt.plot(range(1,ITERATIONS+1), dft_lucb, label='LUCB-RTDP')
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'time [s]')
    plt.legend(loc="upper left")  
    plt.show()
