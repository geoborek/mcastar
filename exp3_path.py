import grid
import numpy as np
from collections import deque
import astar as search
import matplotlib.pyplot as plt
from copy import deepcopy
import ilao
import planning
import time

NUM_SAMPLES = [500] #range(20,101,10)

np.random.seed(2)

env = grid.Environment(12,12, grid.ACTIONS)
env.generate_map(type=0, noise=False, prob=0.02)
env.display()

vfunc_ilao = grid.Vfunction(env)

# ilao.ilao(env.start, env, vfunc_ilao, 0.001)
# print(f"Best action: {vfunc_ilao.get_best_action(env.start)}")
# print(f"Q-value: {vfunc_ilao.evaluate(env.start)}")

# vfunc_ilao.display_vfunc()

state = env.start #grid.State(10,1)
actions = env.get_applicable(state)
q_vals = np.zeros((len(actions), len(NUM_SAMPLES)))
true_costs = np.zeros(len(actions))

vals = np.zeros(len(NUM_SAMPLES))
ilao.ilao(state, env, vfunc_ilao, 0.001)
opt_val = vfunc_ilao.evaluate(state)
print(f"V*({state}) = {opt_val}")

for j, samples in enumerate(NUM_SAMPLES):
    print(f"samples={samples}")
    hfunc = search.Hfunction(env, samples=samples)
    vfunc = grid.Vfunction(env)
    vals[j] = planning.expected_astar(state, env, vfunc, hfunc, noise=0.01)

print(vals)
plt.figure()
plt.plot(NUM_SAMPLES, vals)
# plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Heuristic value")
plt.show()


# for i, a in enumerate(actions):
#     ilao.ilao(state, env, vfunc_ilao, 0.001)
#     true_costs[i] = vfunc_ilao.get_Q_value(state, a)
#     print(f"Q({state}, {a}) = {true_costs[i]}")

# for j, samples in enumerate(NUM_SAMPLES):
#     print(f"samples={samples}")
#     hfunc = search.Hfunction(env, samples=samples)
#     vfunc = grid.Vfunction(env)

#     for i, a in enumerate(actions):
#         all_succs = env.get_successors(state, a)
#         for s in all_succs:
#             planning.expected_astar(s, env, vfunc, hfunc)
#         # print(val)
#         q_vals[i,j] = planning.get_heuristic_Q_value(state, a, env, hfunc)

# print(q_vals)

# plt.figure()
# for i, a in enumerate(actions):
#     plt.plot(NUM_SAMPLES, q_vals[i], label=f"{a}")
# plt.legend()
# plt.xlabel("Number of samples")
# plt.ylabel("Heuristic Q-value")
# plt.show()
    

