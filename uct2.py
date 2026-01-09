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
    def __init__(self, state, env, vfunc, hfunc, parent=None, action=None, init=1):
        self.state = state
        self.parent = parent
        self.action = action
        self.env = env
        self.vfunc = vfunc
        self.hfunc = hfunc

        self.actions = self.env.get_applicable(self.state)
        self.action_hist = init*np.ones(len(self.actions))           # histogram of actions' applications
        self.Q_vals = np.zeros(len(self.actions))
        for i in range(len(self.actions)):
            t, cost = self.env.get_extrem_successor(self.state, self.actions[i], worst=False, beta=0)
            if self.env.is_terminal(t):
                self.Q_vals[i] = cost + self.env.get_terminal_cost(t)
            else:
                self.Q_vals[i] = cost + pl.expected_astar(t, self.env, self.vfunc, hfunc, penalty=True)
                # self.env.heur(t)
        # print(self.state)
        self.untried_actions = 0 #len(self.actions)

        self.visits = init
        # self.V = self.env.heur(self.state)
        self.action_cost = 0.0
       
    def is_terminal_node(self) -> bool:
        return self.env.is_terminal(self.state)

    def is_fully_expanded(self) -> bool:
        return self.untried_actions == 0

    def simulate(self, action, visited):
        next_state, cost = self.env.get_sampled_successor(self.state, self.actions[action])
        self.action_cost = cost
        if next_state in visited:
            child = visited[next_state]
            # self.Q_vals[self.untried_actions] = self.action_cost + child.total_cost/child.visits
            child.action = action
        else:
            child = Node(next_state, self.env, self.vfunc, self.hfunc, parent=self, action=action)
            visited[next_state] = child
        return child

    def expand(self, visited):
        action = self.untried_actions-1
        # self.action_hist[self.untried_actions-1] += 1
        self.untried_actions -= 1
        return self.simulate(action, visited)
    
    def best_action(self, C: float = 1.41):
        return np.argmin([self.Q_vals[i] - C * np.sqrt(np.log(self.visits) / self.action_hist[i]) for i in range(len(self.actions))])

    def best_child(self, visited, C: float = 1.41):
        action = self.best_action(C)
        return self.simulate(action, visited)

    def weight_sum(self, n):
        return n#*(n+1)/2

    def update_pair(self, cost, action):
        n = self.weight_sum(self.action_hist[action])
        # print(self.action_hist)
        if n == 0:
            self.Q_vals[action] = cost
            self.action_hist[action] = 1
        else:            
            self.Q_vals[action] = (self.Q_vals[action]*n + cost)/(n+1)
            self.action_hist[action] += 1

    def backpropagate(self, cost: float, action, next_node, end_node):
        self.visits += 1
        # self.V = (self.V*(self.visits-1) + cost)/self.visits

        if end_node:
            # print("here", self.state, action)
            self.update_pair(cost, action)    
        else:
            # print(next_node.state, next_node.Q_vals, cost)
            self.update_pair(cost+np.min(next_node.Q_vals), action)
            # self.Q_vals[action] = (self.Q_vals[action]*(self.action_hist[action]-1) + self.action_cost + cost)/self.action_hist[action]
        if self.parent:
            self.parent.backpropagate(self.action_cost + cost, self.action, self, False)

    def rollout(self) -> float:
        """
        Random simulation (default rollout policy)
        """
        current = self.state
        c = 0
        while not self.env.is_terminal(current):
            actions = self.env.get_applicable(current)
            # perturbed_Q_vals = np.array([self.vfunc.get_Q_value(current, a) for a in actions]) - np.random.gumbel(0,10,len(actions))
            # i = np.argmin(perturbed_Q_vals)
            # action = actions[i]
            action = random.choice(actions)
            current, cost = self.env.get_sampled_successor(current, action)
            c += cost
        return c+self.env.get_terminal_cost(current)


class SingleAgentUCT:
    def __init__(self, env, vfunc, hfunc, iterations=10000, exploration=1.41):
        self.iterations = iterations
        self.exploration = exploration
        self.env = env
        self.vfunc = vfunc
        self.hfunc = hfunc
        self.visited = {}
        
    def search(self, state):
        root = Node(state, env, vfunc, self.hfunc)
        self.visited[state] = root

        qvals = np.zeros(self.iterations)
        vals = np.zeros(self.iterations)

        for i in range(self.iterations):
            node = root
            if i % 1 == 0:
                print(f"Iteration: {i}")
                # print(node.Q_vals)

            # 1. Selection
            while not node.is_terminal_node() and node.is_fully_expanded():
                print(node.state)
                print(node.actions)
                print(node.Q_vals)
                print(node.best_action(self.exploration))
                node = node.best_child(self.visited, self.exploration)
                # print(node.state)

            # 2. Expansion
            if not node.is_terminal_node():
                print(f"Node to be expanded: {node.state}")
                node = node.expand(self.visited)
                print(f"Expanded node: {node.state}")

            # 3. Simulation
            # cost = node.rollout()
            # cost = pl.expected_astar(node.state, self.env, self.vfunc, hfunc, penalty=True)
            # if cost<1000:
            # print(cost)
            # print(node.state)
            # print(node.Q_vals)

            # 4. Backpropagation
            # if cost != None:
            cost = env.get_terminal_cost(node.state)
            if node.parent:
                # print(node.parent.state, node.state, node.parent.actions[node.action], cost)
                node.parent.backpropagate(cost+node.action_cost, node.action, node, True)
                # print(node.parent.Q_vals)
                print(f"updated Q-vals: {root.Q_vals}")

            qvals[i] = root.Q_vals[5]
            vals[i] = root.Q_vals[3]

        # Return best action
        best = root.best_action(0)
        print(root.state)
        print(root.actions[best])
        print(root.actions)
        print(root.Q_vals)
        # print(root.V)
        # print(self.visited.keys())
        plt.figure()
        # plt.plot(range(self.iterations), vals)
        plt.plot(range(self.iterations), qvals)
        plt.show()

        return root.actions[best]

def planning(agent, env, vfunc):
    state = env.start
    path = [state]
    c = 0

    while not env.is_terminal(state):
        print(state)
        action = agent.search(state) 
        print(action)   
        state, cost = env.get_sampled_successor(state, action, execute=True)
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
    env = grid.Environment(7, 7, grid.ACTIONS)
    env.generate_map(type=3, noise=False, prob=0.02)
    env.display()

    # vfunc = apples.Vfunction(env)
    vfunc = grid.Vfunction(env)
    hfunc = astar.Hfunction(env, samples=10)
    agent = SingleAgentUCT(env, vfunc, hfunc, iterations=30, exploration=0)

    state = env.start #grid.State(0,1)
    root = agent.search(state)
    # planning(agent, env, vfunc)
