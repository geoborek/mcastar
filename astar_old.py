import pqueue as pq
import numpy as np
import grid
import matplotlib.pyplot as plt

def back_propagate(state, parent, action, vfunc, hfunc):
    # path = [state]
    # plan = []
    c = hfunc.evaluate(state)
    s = state
    while s in parent:
        a, cost = action[s]
        c += cost
        # plan.append(a)
        s = parent[s]
        # path.append(s)
        # if s not in parent:
        #     print(s)
        hfunc.update(s, c)
        # vfunc.values[s] = vfunc.get_Q_value(s, a)
        # vfunc.Bellman_update(s)
    # path.reverse()
    # plan.reverse()
    return c

def back_propagate_dead_end(state, parent, action, vfunc, hfunc):
    s = state
    while s in parent:
        a, cost = action[s]
        s = parent[s]
        # vfunc.values[s] = vfunc.get_Q_value(s, a)
        # vfunc.Bellman_update(s)

def astar(state, env, vfunc, hfunc, noise=0.0):
   
    gScore = {}
    parent = {}
    action = {}
    open = pq.PQueue() #hd.heapdict()

    gScore[state] = 0

    # Expand the initial state and compute expected costs for each action
    actions = env.get_applicable(state)
    for a in actions:
        if vfunc.is_safe(state, a):
            h = vfunc.get_Q_value(state, a)
            open[(state, a)] = h + np.random.gumbel(0, noise)
            # print(f"Action: {a}")
            # print(f"Qval: {h}")

    # last_cost = 0
    k = 0
    # print(open.items())
    while open.items():
        ((s, a), fScore) = open.popitem()
        # print(f"State:\n{s}")
        # print(f"Is goal: {env.is_goal(s)}")
        # if env.is_goal(s):
        #     print(f"Cost: {env.get_terminal_cost(s)} {s.rewards()} {hfunc.is_done(s)}")
        # print(f"Action: {a}")
        # print(f"fScore: {fScore}")

        if hfunc.is_done(s):
            # r = np.random.random()
            # if r<np.exp(-k-1):
            #     new_cost = back_propagate(s, parent, action, vfunc, hfunc)
            #     k += 1
            #     last_cost = (last_cost*(k-1) + new_cost)/k
            #     continue
            # else:
            # print(f"State:\n{s}")
            return back_propagate(s, parent, action, vfunc, hfunc)
        
        # for j in range(1):
        t, cost = env.get_sampled_successor(s, a)
        # t, cost = env.get_extrem_successor(s, a, worst=False, beta=0)
        # print(f"Sampled state:\n{t}")
        v = gScore[s] + cost
        if not (t in gScore) or v < gScore[t]:
            gScore[t] = v
            parent[t] = s
            action[t] = (a, cost)
            if env.is_goal(t):
                open[(t, None)] = v + vfunc.evaluate(t)
            elif env.is_terminal(t):
                # vfunc.display_vfunc()
                back_propagate_dead_end(t, parent, action, vfunc, hfunc)
                # vfunc.display_vfunc()
            else:
                for a in env.get_applicable(t):
                    if vfunc.is_safe(t, a):
                        h = vfunc.get_Q_value(t, a)
                        open[(t, a)] = v + h - np.random.gumbel(0, noise)

    return None

class Hfunction:
    def __init__(self, env, samples=10, max_penalty=1000) -> None:
        self.env = env
        self.samples = samples
        self.values = {}
        self.visits = {}
        self.max_penalty = max_penalty

    def evaluate(self, state):
        if self.env.is_terminal(state):
            return self.env.get_terminal_cost(state)
        else:
            return self.values[state]

    def update(self, state, cost):
        if state in self.values.keys():
            n = self.visits[state]
            self.values[state] = (self.values[state]*n + cost)/(n+1)
            self.visits[state] = n+1
        else:
            self.values[state] = cost
            self.visits[state] = 1

    def set_unsolvable(self, state):
        self.values[state] = self.max_penalty
        self.visits[state] = self.samples

    def is_done(self, state):
        return self.env.is_goal(state) or (state in self.visits.keys() and self.visits[state] >= self.samples)
    
    def display_hfunc(self):
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

if __name__ == '__main__':

    print("TODO")

    # edges = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')]
    # cs = [1, 3, 1, 2]
    # hs = { 'a': 3, 'b': 3, 'c': 0, 'd': 0 }

    # def cost(a):
    #     for i in range(len(edges)):
    #         if a == edges[i]:
    #             return cs[i]
            
    # def isGoal(s):
    #     return s == 'd'

    # def getApplicable(s):
    #     return [ e for e in edges if e[0] == s ]

    # def succ(s, a):
    #     return s[1]

    # def heur(s):
    #     return hs[s]

    # plan = astar('a', isGoal, getApplicable, succ, cost, heur)
    # print(plan)