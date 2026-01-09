import numpy as np
from collections import deque
import grid
import astar as search
from tqdm import tqdm
import test_env as test
import lrtdp
import matplotlib.pyplot as plt
import planning

def rollout_action(state, qfunc, state_visited, mode="exploration", exploration=1.41):
    actions = env.get_applicable(state)
    if mode == "uniform":
        return np.random.choice(actions)

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
            h = get_heur_Q_value(state, a, env, vfunc_fixed, hfunc)
            if h < min_h:
                min_h = h
                best_action = a
        if best_action != None:
            return best_action
    a, _ = qfunc.get_best_action(state, state_visited, mode=mode, exploration=exploration)
    return a

def rollout(state, env, qfunc, state_visited, mode="exploration", max_iter=50, exploration=1.41):
    open = deque()

    s = state
    total_cost = 0
    k = 0 
    while not env.is_terminal(s) and k<max_iter:
        a = rollout_action(s, qfunc, state_visited, mode=mode, exploration=exploration)
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
    # print(f"terminal: {terminal_cost}, total: {total_cost}, iterations: {k}")
    return open, terminal_cost, total_cost+terminal_cost

def update(trace, qfunc, state_visited, terminal_cost, init=20):
    cost = terminal_cost
    while len(trace) > 0:
        s, a, c, t = trace.pop()
        if s in state_visited:
            state_visited[s] += 1
        else:
            state_visited[s] = init+1
        cost += c
        qfunc.update(s, a, cost, t, init=init)
    return 0

def mcts(state, env, qfunc, iterations=2000, exploration=0.5):
    state_visited = {}
    pvals = np.zeros(iterations)

    for i in tqdm(range(iterations)):
        trace, terminal_cost, _ = rollout(state, env, qfunc, state_visited, mode="exploration", exploration=exploration)
        update(trace, qfunc, state_visited, terminal_cost)
        if (state, env.actions[0]) in qfunc.values:
            pvals[i] = qfunc.values[(state, env.actions[3])]
            # if i % 5 == 0:
            #     print(terminal_cost)
            #     print(pvals[i])

    best_action, val = qfunc.get_best_action(state,state_visited, mode="best")

    plt.figure()
    plt.plot(range(iterations), pvals)
    plt.show()
    return best_action, val

def estimate(state, env, qfunc, state_visited):
    _, terminal_cost, total_cost = rollout(state, env, qfunc, state_visited, mode="best")
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
        qfunc.update(s, a, cost, t)
    return 0

def brue(state, env, qfunc, iterations=2000):
    state_visited = {}
    pvals = np.zeros(iterations)

    for i in tqdm(range(iterations)):
        trace, terminal_cost, total_cost = rollout(state, env, qfunc, state_visited, mode="uniform")
        brue_update(trace, env, qfunc, state_visited)
        if (state, env.actions[0]) in qfunc.values:
            pvals[i] = qfunc.values[(state, env.actions[0])]

    best_action, val = qfunc.get_best_action(state, state_visited, mode="best")

    # print(qfunc.values)
    plt.figure()
    plt.plot(range(iterations), pvals)
    plt.show()
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

    def weight_sum(self, n):
        return n*(n+1)/2

    def update(self, state, action, cost, target, init=20):
        if (state, action) in self.pair_visited:
            n = self.pair_visited[(state, action)]
            self.values[(state, action)] = (self.values[(state, action)]*self.weight_sum(n) + (n+1)*cost)/self.weight_sum(n+1)
            self.pair_visited[(state, action)] += 1
            self.histograms[(state, action)].update(target, cost)
            # if state==self.env.start and action==self.env.actions[5]:
            #     print(f"Update value: {self.values[(state, action)]}")
            #     print(get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc), cost, init)
        else:            
            h = get_heur_Q_value(state, action, self.env, self.vfunc, self.hfunc)
            self.values[(state, action)] = (h*init + cost)/(init+1)
            self.pair_visited[(state, action)] = init+1
            self.histograms[(state, action)] = Histogram(target, cost)
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
    succs = env.get_successors(state, action, sampled=True)
    res = 0
    for t, c in succs:
        h = heur(t, env, vfunc, hfunc, player=1)
        res += c + h
    return res/len(succs)

def heur(state, env, vfunc, hfunc, player=1):
    return planning.expected_astar(state, env, vfunc, hfunc, penalty=False, player=player)
    # res = search.astar(state, env, vfunc, hfunc)
    # if res != None:
    #     return hfunc.evaluate(state)
    # else:
    # return env.heur(state)

if __name__ == '__main__':

    np.random.seed(3)

    env = grid.Environment(7, 7, grid.ACTIONS)
    env.generate_map(type=3, noise=False, prob=0.05)
    env.display()

    # env = test.env

    vfunc_fixed = grid.Vfunction(env)
    hfunc = search.Hfunction(env, samples=10)
    vfunc = lrtdp.Vfunction(env, vfunc_fixed, hfunc, hfunc)

    qfunc = Qfunction(env, vfunc_fixed, hfunc)

    state = env.start
    # lrtdp.lrtdp(state, env, vfunc, eps=0.01)
    # print(vfunc.get_best_action(state, debug=False))
    # print(vfunc.get_Q_value(state, env.actions[5]))


    a, val = mcts(state, env, qfunc)
    print(a, val)
    for a in env.get_applicable(state):
        if (state, a) in qfunc.values:
            print(a, qfunc.values[(state, a)])
            print(get_heur_Q_value(state, a, env, vfunc_fixed, hfunc))
    qfunc.display_qfunc()