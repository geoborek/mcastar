import numpy as np
import pyspiel
import mcts
import astar as search
import lrtdp
import planning as pl

def state_str2array(str): 
    rows = str.split() 
    return rows

class Environment:
    def __init__(self, game) -> None:
        self.game = pyspiel.load_game(game)
        self.start = self.game.new_initial_state()
        self.rng = np.random.seed(1)
        mcts_params = {"c_puct": 1.4}
        self.mcts = mcts.MCTS(self.game, root_player=1, rng=self.rng, **mcts_params)
        self.successors = {}

    def is_terminal(self, state):
        # print(state)
        return state.is_terminal()

    def is_goal(self, state):
        # return self.heur(state) <= 10
        return state.is_terminal() and self.get_terminal_cost(state) <= 100

    def get_terminal_cost(self, state, player=1):
        return 100-100*state.returns()[player-1]
    
    def get_applicable(self, state):
        return state.legal_actions()
       
    def get_sampled_successor(self, state, action, execute=False):
        if execute:
            s = state
        else:
            s = state.clone()
        s.apply_action(action)
        return s, 0

    def get_successors(self, state, action, sampled=True, num_samples=20, cache=True):
        if cache and (state, action) in self.successors:
            return self.successors[(state, action)]
        elif cache:
            self.successors[(state, action)] = [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
            return self.successors[(state, action)]
        else:
            return [self.get_sampled_successor(state, action, execute=False) for i in range(num_samples)]
   
    def heur(self, state, player=1):
        chrs = state.observation_string(0).split()
        vals = np.sort(np.array([int(chr) for chr in chrs]))
        ck = np.exp2(np.arange(-1,12))
        ck[0] = 0
        hist = np.histogram(vals, bins=ck)[0]
        E = hist[0]
        hist = hist[1:]
        ck = ck[1:]
        mk = 0
        for k in range(1,12):
            s = 0
            for j in range(1,k+1):
                s += hist[j-1]*2**(j-k) 
            if s >= 2:
                mk = k+1

        h = 11-mk
        return h
    
class Vfunction:
    def __init__(self, env) -> None:
        self.env = env
        self.values = {}

    def evaluate(self, state, player=1):
        if self.env.is_terminal(state):
            return self.env.get_terminal_cost(state)
        elif state in self.values.keys():
            return self.values[state]
        else:
            return self.env.heur(state, player)

    def is_safe(self, state, action, threshold=1000):
        # succs = self.env.get_successors(state, action)
        # for s in succs:
        #     print(s)
        #     if self.env.is_terminal(s) and not self.env.is_goal(s):
        #         return False
        #     if self.evaluate(s) > threshold:
        #         return False
        return True

    def get_Q_value(self, state, action, player=1):
        succs = self.env.get_successors(state, action)
        vals = np.array([c+self.evaluate(s, player) for s, c in succs])
        return np.mean(vals)
    
    def get_best_action(self, state, player=1):
        min_h = np.inf
        best_action = None
        for a in self.env.get_applicable(state):
            h = self.get_Q_value(state, a, player=player)
            if h < min_h:
                min_h = h
                best_action = a
        return best_action, min_h

    def Bellman_update(self, state):
        _, val = self.get_best_action(state)
        self.values[state] = val
        return val

def planning(env, vfunc):
    state = env.start
    path = [state]
    c = 0
    while not env.is_terminal(state):
        print(state)
        lrtdp.lrtdp(state, env, vfunc, eps=0.1, num_iter=10)
        a, c = vfunc.get_best_action(state) 
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

if __name__ == '__main__':

    np.random.seed(2)

    env = Environment("2048")

    vfunc_fixed = Vfunction(env)
    hfunc = search.Hfunction(env, samples=1)
    vfunc = lrtdp.Vfunction(env, vfunc_fixed, hfunc, hfunc)

    # val = planning.expected_astar(env.start, env, vfunc, hfunc, noise=0)
    # print(val)

    state = env.start #grid.State(0,2)
    print(state)
    print(env.heur(state))
    vfunc.evaluate(state, player=1)
    # lrtdp.heur(state, env, vfunc_fixed, hfunc, player=1)
    # lrtdp.lrtdp(state, env, vfunc, eps=0.1, num_iter=20)
    # planning(env, vfunc)
