import numpy as np
import math
import random
import grid
import ghosts
import apples
import astar
import planning as pl
from copy import deepcopy
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, env, vfunc, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.env = env
        self.vfunc = vfunc

        self.children = []
        # print(self.state)
        self.untried_actions = deepcopy(self.env.get_applicable(self.state))

        self.visits = 1
        self.total_cost = self.env.heur(self.state)
        self.action_cost = 0.0

    def is_terminal_node(self) -> bool:
        return self.env.is_terminal(self.state)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state, cost = self.env.get_sampled_successor(self.state, action)
        self.action_cost = cost
        child = Node(next_state, self.env, self.vfunc, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, C: float = 1.41):
        return min(
            self.children,
            key=lambda child:
                (child.total_cost / child.visits)
                + C * math.sqrt(math.log(self.visits) / child.visits)
        )

    def backpropagate(self, cost: float):
        self.visits += 1
        self.total_cost += self.action_cost + cost

        if self.parent:
            self.parent.backpropagate(self.action_cost + cost)

    def rollout(self) -> float:
        """
        Random simulation (default rollout policy)
        """
        current = self.state
        c = 0
        while not self.env.is_terminal(current):
            actions = self.env.get_applicable(current)
            # perturbed_Q_vals = np.array([self.vfunc.get_Q_value(current, a) for a in actions]) - np.random.gumbel(0,0.01,len(actions))
            # i = np.argmin(perturbed_Q_vals)
            # action = actions[i]
            action = random.choice(actions)
            current, cost = self.env.get_sampled_successor(current, action)
            c += cost
        return c+self.env.get_terminal_cost(current)


class SingleAgentUCT:
    def __init__(self, env, vfunc, iterations=10000, exploration=1.41):
        self.iterations = iterations
        self.exploration = exploration
        self.env = env
        self.vfunc = vfunc
        
    def search(self, root):
        vals = np.zeros(self.iterations)

        for i in range(self.iterations):

            node = root
            # print(root.is_fully_expanded())
            # 1. Selection
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.best_child(self.exploration)

            # 2. Expansion
            if not node.is_terminal_node():
                print(f"Node to be expanded: {node.state}")
                node = node.expand()
                print(f"Expanded node: {node.state}")

            # 3. Simulation
            cost = node.rollout()
            # cost = pl.expected_astar(node.state, self.env, self.vfunc, hfunc)

            # 4. Backpropagation
            node.backpropagate(cost)

            for node in root.children:
                if node.action.name == 'NE':
                    vals[i] = node.total_cost/node.visits

        # Return most visited action
        # print(root.state, root.children)
        best = max(root.children, key=lambda n: n.visits)
        print([(node.action, node.total_cost/node.visits) for node in root.children])

        plt.figure()
        plt.plot(range(self.iterations), vals)
        plt.show()

        return best

def planning(agent, env, vfunc):
    state = env.start
    path = [state]
    c = 0

    while not env.is_terminal(state):
        print(state)
        root = Node(state, env, vfunc)
        root = agent.search(root) 
        print(root.action)   
        state, cost = env.get_sampled_successor(state, root.action, execute=True)
        path.append(state)
        c += cost

    env.display_path(path)
    if env.is_goal(state):
        print(f"Success! Cost = {c}")
    else:
        print(f"Fail! Cost = {c}")

if __name__ == '__main__':

    # np.random.seed(3)
    # random.seed(2)

    # env = apples.Environment(7,7)
    # env.generate_map()
    env = grid.Environment(3,4, grid.ACTIONS)
    env.generate_map(type=3, noise=False, prob=0.02)
    env.display()

    # vfunc = apples.Vfunction(env)
    vfunc = grid.Vfunction(env)
    hfunc = astar.Hfunction(env, samples=100)

    agent = SingleAgentUCT(env, vfunc, iterations=300)

    root = Node(env.start, env, vfunc)
    root = agent.search(root)
    print(root.action)
    # planning(agent, env, vfunc)
