from collections import deque
import math

class ILAOStar:
    def __init__(self, mdp, epsilon=1e-6):
        self.mdp = mdp
        self.epsilon = epsilon

        # Value function and policy
        self.V = {}
        self.policy = {}
        self.expanded = set()

    def solve(self, s0):
        """Main ILAO* loop"""

        # Initialize
        self.V[s0] = self.mdp.heuristic(s0)
        self.policy[s0] = None

        while True:
            # 1. Compute current solution graph
            solution = self._compute_solution_graph(s0)

            # 2. Find unexpanded tips
            tips = [s for s in solution
                    if s not in self.expanded and not self.mdp.is_terminal(s)]

            # 3. If all expanded, we are done
            if not tips:
                break

            # 4. Expand one tip (change to expand all for batch version)
            s = tips[0]
            self._expand(s)

            # 5. Run value iteration on the partial graph
            self._value_iteration(solution)

        return self.policy, self.V


    # ---------------------------
    # Core subroutines
    # ---------------------------

    def _expand(self, s):
        """Expands a state by generating its successors"""

        for a in self.mdp.actions(s):
            for _, s_next, _ in self.mdp.transitions(s, a):
                if s_next not in self.V:
                    self.V[s_next] = self.mdp.heuristic(s_next)
                    self.policy[s_next] = None

        self.expanded.add(s)


    def _bellman_backup(self, s):
        """One Bellman update"""

        if self.mdp.is_terminal(s):
            return 0, None

        best_value = float("inf")
        best_action = None

        for a in self.mdp.actions(s):
            q = 0
            for p, s_next, cost in self.mdp.transitions(s, a):
                v_next = self.V.get(s_next, self.mdp.heuristic(s_next))
                q += p * (cost + v_next)

            if q < best_value:
                best_value = q
                best_action = a

        return best_value, best_action


    def _compute_solution_graph(self, s0):
        """Follow current greedy policy from start"""

        solution = set()
        stack = deque([s0])

        while stack:
            s = stack.pop()
            if s in solution:
                continue

            solution.add(s)

            # Only follow greedy policy
            a = self.policy.get(s, None)
            if a is not None:
                for _, s_next, _ in self.mdp.transitions(s, a):
                    if s_next not in solution:
                        stack.append(s_next)

        return solution


    def _value_iteration(self, solution):
        """Run VI only on the solution graph"""

        while True:
            delta = 0

            for s in solution:
                if self.mdp.is_terminal(s):
                    continue

                old_v = self.V[s]
                new_v, a = self._bellman_backup(s)

                self.V[s] = new_v
                self.policy[s] = a

                delta = max(delta, abs(old_v - new_v))

            if delta < self.epsilon:
                break
