import grid
import numpy as np
from collections import deque
import astar as search
import matplotlib.pyplot as plt
from copy import deepcopy
import ilao
import planning

NUM_SAMPLES = range(1,20)
DEPTHS = [2,3,4]

np.random.seed(2)

env = grid.Environment(25,25, grid.ACTIONS2)
env.generate_map(type=1, noise=True, prob=0.1)
env.display()

vfunc_ilao = grid.Vfunction(env)

# ilao.ilao(env.start, env, vfunc_ilao, 0.001)
# print(f"Best action: {vfunc_ilao.get_best_action(env.start)}")
# print(f"Q-value: {vfunc_ilao.evaluate(env.start)}")

# vfunc_ilao.display_vfunc()

initial_states = [
    env.start,
    grid.State(11, 1),
    grid.State( 2,22),
    grid.State(10,20),
    grid.State(20,20),
    grid.State(23,14),
    grid.State(12,11),
    grid.State( 6, 8),
    grid.State( 2, 2),
]

env.display_path(initial_states)

for state in initial_states:
    ilao.ilao(state, env, vfunc_ilao, 0.001)

vfunc_ilao.display_best_actions()

all_regrets = np.zeros((len(DEPTHS), len(NUM_SAMPLES)))
for i, depth in enumerate(DEPTHS):
    regrets = []
    for samples in NUM_SAMPLES:
        regret = 0
        for state in initial_states:
            hfunc = search.Hfunction(env, samples=samples)
            vfunc = grid.Vfunction(env)

            a = planning.subtree_ilao(state, env, vfunc, hfunc, depth=depth) 
            # print(a)   
            # hfunc.display_hfunc()

            value_opt = vfunc_ilao.evaluate(state)
            value_action = vfunc_ilao.get_Q_value(state, a)
            regret += 100*(value_action-value_opt)/value_opt
            print(f"State: {state}")
            print(f"Action: {a}")
            print(f"Best action: {vfunc_ilao.get_best_action(state)}")
            print(f"Difference: {value_action-value_opt}") 
        regrets.append(regret/len(initial_states))
    all_regrets[i] = regrets

plt.figure()
for i, d in enumerate(DEPTHS):
    plt.plot(NUM_SAMPLES, all_regrets[i], label=f"depth = {d}")
plt.legend()
plt.xlabel("Number of samples")
plt.ylabel("Regret")
plt.show()

