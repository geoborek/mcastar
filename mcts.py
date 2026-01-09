#!/usr/bin/env python3
"""
mcts_baseline.py

A simple UCT (MCTS) baseline for OpenSpiel games (e.g., "connect_four").

Dependencies:
  - open_spiel (pyspiel) installed via pip
  - numpy

Usage examples:
  # Run a single MCTS search on the initial position of connect_four:
  python mcts_baseline.py --game connect_four --simulations 2000 --report_root

  # Play 10 matches: MCTS (2000 sims) as player 0 vs Random as player 1
  python mcts_baseline.py --game connect_four --simulations 2000 --matches 10 --opponent random

  # Play MCTS vs MCTS
  python mcts_baseline.py --game connect_four --simulations 2000 --matches 5 --opponent mcts
"""

import argparse
import math
import random
import time
from collections import defaultdict

import numpy as np
import pyspiel


class MCTSNode:
    """Node storing statistics for MCTS (UCT)."""

    def __init__(self, parent=None, parent_action=None):
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0  # cumulative sum of rollout returns (w.r.t. root player)
        # (we store totals as returns from perspective of the root player)
        self.is_expanded = False

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """
    Basic UCT MCTS for deterministic perfect-information games (no chance nodes).
    All values/rollouts are measured w.r.t. the root_player (the one for which the search is run).
    """

    def __init__(self, game, root_player, c_puct=1.4, rollout_policy="uniform", rng=None):
        """
        Args:
          game: pyspiel.Game object
          root_player: int, player id the search optimizes for (0 or 1 for connect_four)
          c_puct: exploration constant (standard UCT uses sqrt(2) ~ 1.414)
          rollout_policy: "uniform" (random) - placeholder for future policies
          rng: optional random.Random or numpy.random.Generator
        """
        self.game = game
        self.root_player = root_player
        self.c = c_puct
        self.rollout_policy = rollout_policy
        self.rng = rng or random.Random()

    def _select(self, node, state):
        """
        Traverse the tree until a leaf node to expand or a terminal node.
        Returns (leaf_node, leaf_state) such that:
          - leaf_state is the game state associated with leaf_node
          - leaf_node is not necessarily expanded (we expand it in expand step)
        We operate on cloned states locally (caller ensures state is a clone).
        """
        while True:
            if state.is_terminal():
                return node, state
            cur_player = state.current_player()
            legal = state.legal_actions()

            # If node is not expanded yet, stop here
            if not node.is_expanded:
                return node, state

            # Choose child with highest UCT score
            best_action, best_child = None, None
            best_score = -float("inf")
            parent_N = node.visit_count
            for a in legal:
                child = node.children.get(a)
                if child is None or child.visit_count == 0:
                    # force exploration of unvisited children
                    uct = float("inf")
                else:
                    q = child.q_value()
                    uct = q + self.c * math.sqrt(math.log(parent_N + 1) / child.visit_count)
                if uct > best_score:
                    best_score = uct
                    best_action = a
                    best_child = child

            # If best_child is None (unseen action), expand by creating child and apply action
            if best_action not in node.children:
                # create child placeholder and return it for expansion
                node.children[best_action] = MCTSNode(parent=node, parent_action=best_action)
                state.apply_action(best_action)
                return node.children[best_action], state

            # otherwise descend
            state.apply_action(best_action)
            node = best_child

    def _expand(self, node, state):
        """
        Create children for all legal actions at state and mark node expanded.
        We do not initialize statistics for children beyond creating nodes.
        """
        if state.is_terminal():
            node.is_expanded = True
            return
        legal = state.legal_actions()
        for a in legal:
            if a not in node.children:
                node.children[a] = MCTSNode(parent=node, parent_action=a)
        node.is_expanded = True

    def _rollout(self, state):
        """Perform a random rollout from state until terminal. Return return[root_player]."""
        s = state.clone()
        while not s.is_terminal():
            legal = s.legal_actions()
            a = self.rng.choice(legal)
            s.apply_action(a)
        returns = s.returns()
        # returns is an array-like of length num_players
        return returns[self.root_player]

    def _backpropagate(self, node, value):
        """
        Backpropagate rollout value (value is return w.r.t. root_player).
        We accumulate value directly (no sign flip) because all values are measured w.r.t. root_player.
        """
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def search(self, root_state, num_simulations=1000, time_limit_s=None):
        """
        Run MCTS search from root_state (a pyspiel.State clone).
        Returns the root node after search.

        Args:
          root_state: pyspiel.State (should be a clone if caller wants to reuse original)
          num_simulations: int, maximum number of simulations (ignored if time_limit_s is set)
          time_limit_s: float seconds, optional soft time limit (overrides num_simulations if set)
        """
        root_node = MCTSNode(parent=None, parent_action=None)
        # Expand root node (children placeholders) to allow proper selection
        self._expand(root_node, root_state)

        sims = 0
        start_time = time.time()
        while True:
            if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
                break
            if num_simulations is not None and sims >= num_simulations:
                break

            # clone root each simulation
            state = root_state.clone()
            node = root_node

            # SELECTION & EXPANSION
            leaf, leaf_state = self._select(node, state)
            # if leaf_state is terminal we can directly get reward
            if leaf_state.is_terminal():
                value = leaf_state.returns()[self.root_player]
                # ensure node is treated as expanded terminal
                leaf.is_expanded = True
            else:
                # expand leaf
                self._expand(leaf, leaf_state)
                # perform rollout from leaf_state
                value = self._rollout(leaf_state)

            # BACKPROPAGATION
            self._backpropagate(leaf, value)
            sims += 1

        return root_node

    def best_action(self, root_state, num_simulations=1000, time_limit_s=None, temp=0.0):
        """
        Run search and return the selected action.
        Selection after search uses visit counts. If temp==0 choose argmax visits (greedy).
        If temp>0 produce a probability distribution proportional to visit_count^(1/temp).
        """
        root_node = self.search(root_state, num_simulations=num_simulations, time_limit_s=time_limit_s)
        # choose child by visits
        legal = root_state.legal_actions()
        visits = []
        for a in legal:
            child = root_node.children.get(a)
            visits.append((a, child.visit_count if child is not None else 0))

        if temp == 0.0:
            # pick argmax visits (break ties randomly)
            max_v = max(v for (_, v) in visits)
            best = [a for (a, v) in visits if v == max_v]
            return random.choice(best), root_node
        else:
            arr = np.array([v for (_, v) in visits], dtype=float)
            # prevent all zeros
            if arr.sum() == 0:
                probs = np.ones_like(arr) / len(arr)
            else:
                probs = arr ** (1.0 / temp)
                probs = probs / probs.sum()
            choice = np.random.choice(len(visits), p=probs)
            return visits[choice][0], root_node


def play_match(game, mcts_params, opponent="random", seed=None, verbose=False):
    """
    Play a single match on `game`:
      - player 0: MCTS with params mcts_params (dict)
      - player 1: opponent ("random" | "mcts") - if "mcts", uses same params
    Returns: final returns list
    """
    rng = random.Random(seed)
    state = game.new_initial_state()

    # Precreate MCTS instances
    mcts0 = MCTS(game, root_player=0, rng=rng, **mcts_params)
    mcts1 = MCTS(game, root_player=1, rng=rng, **mcts_params) if opponent == "mcts" else None

    while not state.is_terminal():
        cur = state.current_player()
        if cur == 0:
            # MCTS for player 0: search w.r.t. root_player=0
            action, _root_node = mcts0.best_action(state, num_simulations=2000)
        else:
            if opponent == "random":
                legal = state.legal_actions()
                action = rng.choice(legal)
            elif opponent == "mcts":
                action, _root_node = mcts1.best_action(state, num_simulations=2000)
            else:
                raise ValueError("Unknown opponent: %s" % opponent)
        state.apply_action(action)
        if verbose:
            print("Player", cur, "played", action)

    return state.returns()


def run_matches(game_name="connect_four", num_matches=5, mcts_simulations=2000, opponent="random", seed=0):
    game = pyspiel.load_game(game_name)
    print("Loaded game:", game_name)
    print("Num players:", game.num_players(), "Game type:", game.get_type())

    mcts_params = {"c_puct": 1.4}
    # mcts_params = {"num_simulations": mcts_simulations, "c_puct": 1.4}

    results = []
    for i in range(num_matches):
        r = play_match(game, mcts_params, opponent=opponent, seed=seed + i)
        results.append(r)
        print(f"Match {i}: returns={r}")
    # summarize: for zero-sum two-player, we can show wins/draws/losses for player 0
    p0_scores = [r[0] for r in results]
    wins = sum(1 for s in p0_scores if s > 0)
    draws = sum(1 for s in p0_scores if s == 0)
    losses = sum(1 for s in p0_scores if s < 0)
    print(f"Summary for player0 vs {opponent}: wins={wins}, draws={draws}, losses={losses}")
    return results


def report_root_visit_distribution(game_name="connect_four", num_simulations=2000):
    game = pyspiel.load_game(game_name)
    root = game.new_initial_state()
    mcts = MCTS(game, root_player=0, rng=random.Random(), c_puct=1.4)
    root_node = mcts.search(root, num_simulations=num_simulations)
    legal = root.legal_actions()
    info = []
    for a in legal:
        child = root_node.children.get(a)
        visits = child.visit_count if child is not None else 0
        q = child.q_value() if child is not None and child.visit_count > 0 else None
        info.append((a, visits, q))
    print("Root visit statistics (action, visits, q):")
    for a, v, q in info:
        print(f"  action={a:2d} visits={v:6d} q={q}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="connect_four", help="OpenSpiel game name")
    parser.add_argument("--simulations", type=int, default=2000, help="MCTS simulations per move")
    parser.add_argument("--matches", type=int, default=0, help="If >0, play that many matches (MCTS vs opponent)")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "mcts"], help="Opponent type")
    parser.add_argument("--report_root", action="store_true", help="Run search from root and print visit distribution")
    args = parser.parse_args()

    if args.matches > 0:
        run_matches(game_name=args.game, num_matches=args.matches, mcts_simulations=args.simulations, opponent=args.opponent)
    elif args.report_root:
        report_root_visit_distribution(game_name=args.game, num_simulations=args.simulations)
    else:
        # single-step demonstration: show selected action at root
        game = pyspiel.load_game(args.game)
        root = game.new_initial_state()
        mcts = MCTS(game, root_player=0, rng=random.Random(), c_puct=1.4)
        action, root_node = mcts.best_action(root, num_simulations=args.simulations)
        print("Selected action at root (after %d sims): %s" % (args.simulations, action))
        # print visit distribution
        legal = root.legal_actions()
        for a in legal:
            child = root_node.children.get(a)
            visits = child.visit_count if child is not None else 0
            print(f"Action {a:2d} visits={visits}")

if __name__ == "__main__":
    main()
