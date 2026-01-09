import grid
import numpy as np
from collections import deque
import astar as search
import matplotlib.pyplot as plt
from copy import deepcopy
import ghosts
import apples
import uct
import ilao
import openspiel as sp

MAX_PENALTY = 1000

def expected_astar(state, env, vfunc, hfunc, penalty=True, noise=0.0, player=1):
    if hfunc.is_done(state):
        return hfunc.evaluate(state)
    else:
        remaining_samples = hfunc.samples
        if state in hfunc.visits.keys():
            remaining_samples -= hfunc.visits[state]
        for i in range(remaining_samples):
            res = search.astar(state, env, vfunc, hfunc, noise)
            # if state not in hfunc.values.keys():
            #     print("here2")
            #     print(state)
                # print(hfunc.values.keys())
            # print(i, res)
            if res == None:
                # print("There is no safe policy!")
                if penalty:
                    hfunc.set_unsolvable(state)
                    return MAX_PENALTY
                else:
                    return env.heur(state)
    return hfunc.evaluate(state)

def sequential_halving(state, env, vfunc, hfunc):
    NUM_ACTION_TRIALS = 5 

    actions = env.get_applicable(state) 
    expected_costs = np.zeros(len(actions))

    for i, a in enumerate(actions):
        expected_cost = 0
        for j in range(NUM_ACTION_TRIALS):
            t, cost = env.get_sampled_successor(state, a)
            if env.is_terminal(t):
                c = cost + env.get_terminal_cost(t)
            else:
                c = cost + expected_astar(t, env, vfunc, hfunc)
            expected_cost += c
        expected_costs[i] = expected_cost

    # print(actions)
    # print(expected_costs/NUM_ACTION_TRIALS)
    best = np.argmin(expected_costs)
    return actions[best]

def build_subtree(root, env, vfunc, hfunc, depth=2):
    closed = deque()
    open_depth = deque()
    open_closed = deque()

    open_depth.append((root, 1))
    open_closed.append(root)
    values = {}
    succs = {}

    while len(open_depth) > 0:
        state, d = open_depth.pop()
        closed.append(state)
        succs[state] = []

        if env.is_terminal(state):
            values[state] = env.get_terminal_cost(state)
        elif d == depth:
            values[state] = expected_astar(state, env, vfunc, hfunc)
        else:
            values[state] = expected_astar(state, env, vfunc, hfunc)
            # if hfunc.is_done(state):
            #     values[state] = hfunc.evaluate(state)
            # else:
            #     values[state] = vfunc.evaluate(state)
            actions = env.get_applicable(state)
            for a in actions:
                for s in env.get_successors(state, a):
                    if not s in open_closed:
                        open_depth.append((s, d+1))
                        open_closed.append(s)
                    succs[state].append(s)
    return closed, values, succs

def get_heuristic_Q_value(state, action, values, env):
    all_succs = env.get_successors(state, action)
    vals = np.array([values[t] for t in all_succs])
    ps = action.distribution
    q = np.sum((vals + action.costs) * ps) #- np.random.gumbel(0,1)
    return q

def subtree_ilao(state, env, vfunc, hfunc, epsilon=0.01, depth=3):
    tree, values, succs = build_subtree(state, env, vfunc, hfunc, depth)
    reversed_tree = reversed(tree)
    best_action = None
    res = np.inf
    # new_values = deepcopy(values)
    # print(tree)
    # print(succs)
    while res > epsilon:
        res = 0
        count = 0
        for s in reversed_tree:
            if len(succs[s])>0:
                new_val = np.inf
                actions = env.get_applicable(s)
                for a in actions:
                    q = get_heuristic_Q_value(s, a, values, env)
                    # print(f"State: {s}, Action: {a}, Q-value: {q}")
                    if q < new_val:
                        new_val = q
                        # if s == state:
                        #     best_action = a
                delta = abs(new_val-values[s])
                if delta > res:
                    res = delta
                values[s] = new_val
                # print(f"State value: {new_val}")
            else:
                # print(f"State: {s}, Heur: {values[s]}")
                count += 1
        # print(f"Residual: {res}")
        # print(f"Count: {count}")
        # values = deepcopy(new_values)    

    new_val = np.inf
    actions = env.get_applicable(state)
    for a in actions:
        all_succs = env.get_successors(state, a)
        vals = np.array([values[t] for t in all_succs])
        ps = a.distribution
        q = np.sum((vals + a.costs) * ps) #- np.random.gumbel(0,1)
        # print(f"q={q}, new_val={new_val}, cond={q < new_val}")
        if q < new_val:
            new_val = q
            best_action = a
        # print(f"Action: {a}")
        # print(f"Q-value: {q}")

    # im_val = np.zeros(env.map.shape)

    # for y in range(im_val.shape[0]):
    #     for x in range(im_val.shape[1]):
    #         s = grid.State(x, y)
    #         if env.is_terminal(s):
    #             im_val[y, x] = 0
    #         # else:
    #         #     im_val[y, x] = self.evaluate(s) 
    #         if s in values.keys():
    #             im_val[y,x] = values[s]

    # plt.figure(2)
    # plt.imshow(im_val, norm='linear')
    # plt.colorbar()
    # plt.show()

    return best_action

def planning(env, vfunc, hfunc):
    state = env.start
    path = [state]
    c = 0
    while not env.is_terminal(state):
        print(state)
        a = sequential_halving(state, env, vfunc, hfunc) 
        print(a)   
        state, cost = env.get_sampled_successor(state, a, execute=True)

        path.append(state)
        c += cost

    # env.display_path(path)
    print(state)
    if env.is_goal(state):
        print(f"Success! Cost = {c}")
    else:
        print(f"Fail! Cost = {c}")

def planning_ilao(env, vfunc, hfunc):
    state = env.start
    path = [state]
    c = 0
    while not env.is_terminal(state):
        print(state)
        a = subtree_ilao(state, env, vfunc, hfunc) 
        print(a)   
        state, cost = env.get_sampled_successor(state, a)
        path.append(state)
        c += cost

    env.display_path(path)
    if env.is_goal(state):
        print(f"Success! Cost = {c}")
    else:
        print(f"Fail! Cost = {c}")

if __name__ == '__main__':

    # np.random.seed(2)

    env = sp.Environment("connect_four")
    # env = grid.Environment(15,15, grid.ACTIONS)
    # env.generate_map(type=0, noise=False, prob=0.02)
    # env.display()


    # vfunc_ilao = grid.Vfunction(env)
    # ilao.ilao(env.start, env, vfunc_ilao, 0.001)
    # opt_val = vfunc_ilao.evaluate(env.start)
    # print(f"V*({env.start}) = {opt_val}")

    vfunc = sp.Vfunction(env)
    hfunc = search.Hfunction(env, samples=1)

    state = env.start
    # for a in [0,2,1,5]:
    #     state.apply_action(a)

    # val = expected_astar(state, env, vfunc, hfunc, noise=0)
    # print(val)
    # best = sequential_halving(env.start, env, vfunc)
    # print(best)
    # vfunc.display_best_actions()

    planning(env, vfunc, hfunc)
    # planning_ilao(env, vfunc, hfunc)
    # hfunc.display_hfunc()

    # agent = uct.SingleAgentUCT(env, vfunc, iterations=100000)

    # root = uct.Node(env.start, env, vfunc)
    # root = agent.search(root)
    # print(root.action)
    # uct.planning(agent, env, vfunc)