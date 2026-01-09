import numpy as np

TERMINAL_COST = 1

class Action:
    def __init__(self, name, source, targets, costs, distribution):
        self.name = name
        self.source = source
        self.targets = targets
        self.costs = costs
        self.distribution = distribution

    def __repr__(self) -> str:
        return self.name 
    
    def sample(self):
        t = np.random.choice(self.targets, p=self.distribution)
        return t, self.costs[self.targets.index(t)]
    
    def is_applicable(self, state):
        return state == self.source
        
class Environment:
    def __init__(self, states, start, actions, goals, terminals):
        self.states = states
        self.start = start
        self.actions = actions
        self.goals = goals
        self.terminals = terminals
        self.successors = {}

    def is_terminal(self, state):
        return state in self.terminals or state in self.goals
    
    def is_goal(self, state):
        return state in self.goals

    def get_terminal_cost(self, state, player=1):
        if self.is_goal(state):
            return 0
        else:
            return TERMINAL_COST

    def get_applicable(self, state):
        actions = []
        for a in self.actions:
            if a.is_applicable(state):
                actions.append(a)
        return actions 

    def get_sampled_successor(self, state, action, execute=False):
        t, cost = action.sample()
        return t, cost

    def get_successors(self, state, action, sampled=False, num_samples=20, cache=True):
        if sampled:
            if cache and (state, action) in self.successors:
                return self.successors[(state, action)]
            elif cache:
                self.successors[(state, action)] = [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
                return self.successors[(state, action)]
            else:
                return [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
        else:
            return action.targets
        
    def heur(self, state):
        return 0

STATES = [0, 1, 2, 3, 4]
TERMINALS = [4]
GOALS = [2, 3]

ACTIONS = [Action("a", 0, [1], [0.001], [1]),
           Action("b", 0, [2], [0.003], [1]),
           Action("c", 1, [3], [0.001], [1]),
           Action("d", 1, [4], [0.001], [1])]

env = Environment(STATES, 0, ACTIONS, GOALS, TERMINALS)