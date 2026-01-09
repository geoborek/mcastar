import grid
import numpy as np
from collections import deque
import astar as search
import matplotlib.pyplot as plt
from copy import deepcopy
import ilao
import planning
import time

NUM_SAMPLES = range(5,31,5)
DEPTHS = [2,3,4,5]
TIME_OUT = 60

np.random.seed(2)

env = grid.Environment(25,25, grid.ACTIONS)
env.generate_map(type=0, noise=True, prob=0.05)
env.display()

vfunc_ilao = grid.Vfunction(env)

# ilao.ilao(env.start, env, vfunc_ilao, 0.001)
# print(f"Best action: {vfunc_ilao.get_best_action(env.start)}")
# print(f"Q-value: {vfunc_ilao.evaluate(env.start)}")

# vfunc_ilao.display_vfunc()

all_regrets = np.zeros((len(DEPTHS), len(NUM_SAMPLES)))

for i, d in enumerate(DEPTHS):
    regrets = []
    print(f"depth={d}")
    for samples in NUM_SAMPLES:
        print(f"samples={samples}")
        hfunc = search.Hfunction(env, samples=samples)
        vfunc = grid.Vfunction(env)

        # a = planning.expected_astar(grid.State(2,22), env, vfunc, hfunc)
        # a = planning.subtree_ilao(grid.State(2,22), env, vfunc, hfunc, depth=3) 
        # print(a)
        # vfunc.get_best_action(grid.State(2,22), debug=True)
        # print(vfunc.evaluate(grid.State(2,22)))
        # hfunc.display_hfunc()

        state = env.start
        path = [state]
        c = 0
        cumulative_regret = 0
        time_start = time.time()
        while not env.is_terminal(state) and time.time() < time_start + TIME_OUT:
            ilao.ilao(state, env, vfunc_ilao, 0.001)
            a = planning.subtree_ilao(state, env, vfunc, hfunc, depth=d) 

            value_opt = vfunc_ilao.evaluate(state)
            value_action = vfunc_ilao.get_Q_value(state, a)
            cumulative_regret += 100*(value_action-value_opt)/value_opt

            # if value_action-value_opt > 0.1:
            #     print(state)
            #     print(f"Action: {a}")   
            #     print(f"Best action: {vfunc_ilao.get_best_action(state)}")
            #     print(f"Difference: {value_action-value_opt}") 
                # vfunc.get_best_action(state, debug=True)

            state, cost = env.get_sampled_successor(state, a)
            path.append(state)
            c += cost

        if time.time() - time_start > TIME_OUT:
            print(f"Timeout: depth={d}, samples={samples}")
        # env.display_path(path)
        regrets.append(cumulative_regret)
        print(f"Cumulative regret: {cumulative_regret}")
        if env.is_goal(state):
            print(f"Success! Cost = {c}")
        else:
            print(f"Fail! Cost = {c}")
    all_regrets[i] = regrets

plt.figure()
for i, d in enumerate(DEPTHS):
    plt.plot(NUM_SAMPLES, all_regrets[i], label=f"depth = {d}")
plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Cumulative regret")
plt.show()
    

# vfunc_ilao.display_best_actions()

# all_regrets = np.zeros((len(DEPTHS), len(NUM_SAMPLES)))
# for i, depth in enumerate(DEPTHS):
#     regrets = []
#     for samples in NUM_SAMPLES:
#         regret = 0
#         for state in initial_states:
#             hfunc = search.Hfunction(env, samples=samples)
#             vfunc = grid.Vfunction(env)

#             a = planning.subtree_ilao(state, env, vfunc, hfunc, depth=depth) 
#             # print(a)   
#             # hfunc.display_hfunc()

#             value_opt = vfunc_ilao.evaluate(state)
#             value_action = vfunc_ilao.get_Q_value(state, a)
#             regret += 100*(value_action-value_opt)/value_opt
#             print(f"State: {state}")
#             print(f"Action: {a}")
#             print(f"Best action: {vfunc_ilao.get_best_action(state)}")
#             print(f"Difference: {value_action-value_opt}") 
#         regrets.append(regret/len(initial_states))
#     all_regrets[i] = regrets


