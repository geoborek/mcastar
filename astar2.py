# from __future__ import annotations
import pqueue as pq
import numpy as np
from collections import deque
import grid
import matplotlib.pyplot as plt
from dataclasses import dataclass
from frozenlist import FrozenList
from tqdm import tqdm

@dataclass(eq=True, frozen=True)
class Node:
   state: grid.State
#    history: FrozenList[grid.ShiftAction] 
   gScore: float

def back_propagate(node, parent, applied_action, updated, qfunc):
    _, c = qfunc.evaluate_state(node)
    # print(node in parent)
    while node not in updated and node in parent:
        updated.add(node)
        parent_node = parent[node]
        (action, action_cost) = applied_action[node]
        qfunc.update(parent_node, node, action, action_cost)
        c += action_cost
        node = parent_node
    return c


def astar(node, env, vfunc, qfunc):
    parent = {}
    applied_action = {}
    visited = set()
    open = pq.PQueue() 
    # to_update = deque()
    updated = set()

    # Expand the initial state and compute expected costs for each action
    actions = env.get_applicable(node.state)
    for a in actions:
        if vfunc.is_safe(node.state, a):
            h = vfunc.get_Q_value(node.state, a)          
            open[(node, a)] = h 
            # print(f"Action: {a}")
            # print(f"Qval: {h}")

    k = 0
    res = np.inf
    first_goal = False
    while open.items():
        ((node, action), fScore) = open.popitem()
        # history = node.history
        visited.add(node)
        if node.gScore >= 1.1*res: 
            return res

        if qfunc.is_done(node.state, action):
        #     # print(f"State:\n{s}")
            # print(node)
            if not first_goal:
                res = back_propagate(node, parent, applied_action, updated, qfunc)
                first_goal = True
            else:
                back_propagate(node, parent, applied_action, updated, qfunc)
            # print(f"Goal: {res}")
            continue
        #     # return res
        
        t, cost = env.get_sampled_successor(node.state, action)
        # print(f"Sampled state:\n{t}")
        v = node.gScore + cost
        # new_history = list(history)
        # new_history.append(action)
        # history = FrozenList(new_history)
        # history.freeze()
        next_node = Node(t, v)
        parent[next_node] = node 
        applied_action[next_node] = (action, cost)
        if next_node not in visited or v < next_node.gScore:
            if env.is_goal(t):
                open[(next_node, None)] = v + vfunc.evaluate(t)
            else:
                for a in env.get_applicable(t):
                    if vfunc.is_safe(t, a):
                        h = vfunc.get_Q_value(t, a)
                        open[(next_node, a)] = v + h 

    return res

class Qfunction:
    def __init__(self, env, vfunc, samples=10) -> None:
        self.env = env
        self.vfunc = vfunc
        self.samples = samples
        self.values = {}
        self.visits = {}

    def evaluate(self, node, action):
        if self.env.is_terminal(node.state):
            return self.env.get_terminal_cost(node.state)
        elif (node, action) in self.values:
            return self.values[(node, action)]/self.weight_sum(self.visits[(node, action)])
        else:
            return self.vfunc.get_Q_value(node.state, action)

    def evaluate_state(self, node):
        if self.env.is_terminal(node.state):
            return None, self.env.get_terminal_cost(node.state)
        else:
            out = np.inf
            best_action = None
            actions = self.env.get_applicable(node.state)
            for a in actions:
                val = self.evaluate(node, a)
                if (node, a) in self.values and val < out:
                    out = val
                    best_action = a
            return best_action, out 

    def weight_sum(self, n):
        return n#*(n+1)/2

    def update_pair(self, node, action, cost):
        if (node, action) in self.values:
            n = self.visits[(node, action)]
            self.values[(node, action)] = self.values[(node, action)] + cost
            self.visits[(node, action)] = n+1
        else:
            self.values[(node, action)] = cost
            self.visits[(node, action)] = 1

    def update(self, node, next_node, action, action_cost):
        # out = True
        if self.env.is_terminal(next_node.state):
            _, c = self.evaluate_state(next_node)
            self.update_pair(node, action, action_cost+c)
            # if state == grid.State(0,2):
            #     print(state, next_state, action, action_cost+c)
            #     print(qfunc.values[(state, action)])
            # return out
        else:
            actions = self.env.get_applicable(next_node.state)
            res = np.inf
            # best_action = None
            for a in actions:
                if (next_node, a) in self.values:
                    val = self.evaluate(next_node, a)
                    if val < res:
                        res = val
                        # best_action = a
            # if state == grid.State(0,2):
            #     print(state, next_state, action, action_cost, res)
            #     print(qfunc.values[(state, action)])
            # if next_action == best_action:
            if res < np.inf:
                self.update_pair(node, action, action_cost+res)


    # def set_unsolvable(self, state):
    #     self.values[state] = self.max_penalty
    #     self.visits[state] = self.samples

    def is_done(self, state, action):
        return self.env.is_goal(state) #or ((state, action) in self.visits.keys() and self.visits[(state, action)] >= self.samples)
    
    def is_state_done(self, state):
        if self.env.is_goal(state):
            return True
        else:
            out = True
            actions = self.env.get_applicable(state)
            for a in actions:
                if not self.is_done(state, a):
                    out = False
                    break
            return out 
        
    def display_qfunc(self):
        im_val = np.zeros(self.env.map.shape)

        for y in range(im_val.shape[0]):
            for x in range(im_val.shape[1]):
                s = grid.State(x, y)
                if self.env.is_terminal(s):
                    im_val[y, x] = 0
                else:
                    _, val = self.evaluate_state(s) 
                    im_val[y, x] = val

        plt.figure()
        plt.imshow(im_val, norm='linear')
        plt.colorbar()
        plt.show()

def expected_astar(state, env, vfunc, qfunc):
    if qfunc.is_state_done(state):
        return qfunc.evaluate_state(state) #[qfunc.evaluate(state, a) for a in env.get_applicable(state)] 
    else:
        remaining_samples = qfunc.samples
        if state in qfunc.visits.keys():
            remaining_samples -= qfunc.visits[state]
        for i in range(remaining_samples):
            res = astar(state, env, vfunc, qfunc)
            if res == None:
                print("There is no safe policy!")
                # if penalty:
                #     hfunc.set_unsolvable(state)
                #     return MAX_PENALTY
                # else:
                #     return env.heur(state)
    return qfunc.evaluate(state)

if __name__ == '__main__':

    np.random.seed(2)

    env = grid.Environment(5,5, grid.ACTIONS)
    env.generate_map(type=3, noise=False, prob=0.03)

    env.display()

    vfunc = grid.Vfunction(env)
    qfunc = Qfunction(env, vfunc, samples=1)

    history = FrozenList()
    history.freeze()
    root = Node(env.start, 0)

    ITER = 1000
    vals = np.zeros(ITER)
    for i in tqdm(range(ITER)):
        res = astar(root, env, vfunc, qfunc)
        vals[i] = qfunc.evaluate(root, env.actions[5])
        # print(res)
    print([(a, qfunc.evaluate(root, a)) for a in env.get_applicable(env.start) if (root, a) in qfunc.values]) 
    # print([(a, qfunc.evaluate(grid.State(0,1), a)) for a in env.get_applicable(grid.State(0,1)) if (grid.State(0,1), a) in qfunc.values]) 
    # print(vfunc.evaluate(env.start))

    # qfunc.display_qfunc()

    plt.figure()
    plt.plot(range(ITER), vals)
    plt.show()

    