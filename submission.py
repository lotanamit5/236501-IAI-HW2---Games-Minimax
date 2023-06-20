import math
import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def utility(env: WarehouseEnv, taxi_id: int):
    if env.done():
        agent = env.get_robot(taxi_id)
        other = env.get_robot((taxi_id + 1) % 2)
        if agent.credit > other.credit:
            return math.inf
        if agent.credit < other.credit:
            return -math.inf
        else:
            return -50
    return None


def smart_heuristic(env: WarehouseEnv, taxi_id: int):
    agent = env.get_robot(taxi_id)

    if agent.package is not None:
        return (agent.credit * 1000) \
            + (manhattan_distance(agent.package.position, agent.package.destination)) \
            - manhattan_distance(agent.position, agent.package.destination) \
            + 100

    available_packages = [p for p in env.packages if p.on_board]
    p = sorted(available_packages, key=lambda p: manhattan_distance(agent.position, p.position))[0]

    return (agent.credit * 1000) - manhattan_distance(agent.position, p.position)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class RBAgent(Agent):
    def __init__(self):
        self.time_limit = None
        self.start_time = None

    def check_time(self, epsilon=5e-2):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_limit - epsilon:
            raise TimeoutError

    def heuristic(self, env: WarehouseEnv, agent_id: int):
        if env.done():
            return utility(env, agent_id)
        return smart_heuristic(env, agent_id) - smart_heuristic(env, (agent_id + 1) % 2)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

        operator = 'park'
        D = 0
        while True:
            try:
                operator = self.search(env, agent_id, D)
                D += 1

            except TimeoutError:
                return operator
    
    def search(self, env: WarehouseEnv, agent_id: int, depth:int):
        raise NotImplementedError


class AgentMinimax(RBAgent):
    # TODO: section b : 1
    def search(self, env: WarehouseEnv, agent_id: int, depth: int):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        other_id = (agent_id + 1) % 2
        children_heuristics = [self.RB_Minimax(child, agent_id, depth, turn=other_id) for child in children]
        max_heuristic = max(children_heuristics)
        possible_moves = [i for i, c in enumerate(children_heuristics) if c == max_heuristic]
        operator = operators[random.choice(possible_moves)]
        return operator
    
    def RB_Minimax(self, env: WarehouseEnv, agent_id, depth, turn):
        self.check_time()

        if env.done() or depth == 0:
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]

        for child, op in zip(children, operators):
            child.apply_operator(turn, op)

        if turn == agent_id:
            curr_max = -math.inf
            for c in children:
                v = self.RB_Minimax(c, agent_id, depth - 1, (turn + 1) % 2)
                curr_max = max(v, curr_max)
            return curr_max

        else:
            curr_min = math.inf
            for c in children:
                v = self.RB_Minimax(c, agent_id, depth - 1, (turn + 1) % 2)
                curr_min = min(v, curr_min)
            return curr_min


class AgentAlphaBeta(RBAgent):
    # TODO: section c : 1
    def search(self, env: WarehouseEnv, agent_id, depth):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        other_id = (agent_id + 1) % 2
        children_heuristics = [
            self.RB_AlphaBeta(child, agent_id, depth, turn=other_id, alpha=-math.inf, beta=math.inf)
            for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        operator = operators[index_selected]
        return operator

    def RB_AlphaBeta(self, env: WarehouseEnv, agent_id, depth, turn, alpha, beta):
        self.check_time()

        if env.done() or depth == 0:
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]

        for child, op in zip(children, operators):
            child.apply_operator(turn, op)

        if turn == agent_id:
            curr_max = -math.inf
            for c in children:
                v = self.RB_AlphaBeta(c, agent_id, depth - 1, (turn + 1) % 2, alpha, beta)
                curr_max = max(v, curr_max)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max

        else:
            curr_min = math.inf
            for c in children:
                v = self.RB_AlphaBeta(c, agent_id, depth - 1, (turn + 1) % 2, alpha, beta)
                curr_min = min(v, curr_min)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -math.inf
            return curr_min


class AgentExpectimax(RBAgent):
    # TODO: section d : 1
    def search(self, env: WarehouseEnv, agent_id, depth):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        other_id = (agent_id + 1) % 2
        children_heuristics = \
            [self.RB_Expectimax(child, agent_id, depth, turn=other_id)
            for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        operator = operators[index_selected]
        return operator
    
    def RB_Expectimax(self, env: WarehouseEnv, agent_id, depth, turn):
        self.check_time()

        if env.done() or depth == 0:
            return self.heuristic(env, agent_id)
        
        other_id = (turn + 1) % 2
        
        operators = env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]

        for child, op in zip(children, operators):
            child.apply_operator(turn, op)

        if turn == agent_id:
            curr_max = -math.inf
            for c in children:
                v = self.RB_Expectimax(c, agent_id, depth - 1, other_id)
                curr_max = max(v, curr_max)
            return curr_max

        else:
            probs = [1] * len(operators)
            for i, (c, op) in enumerate(zip(children, operators)):
                stations = [station.position for station in c.charge_stations]
                if op != 'charge' and c.get_robot(other_id).position in stations:
                    probs[i] += sum([c.get_robot(other_id).position == s for s in stations])
            total_sum = sum(probs)
            probs = [p / total_sum for p in probs]
            
            v = 0
            for c, p in zip(children, probs):
                v += p * self.RB_Expectimax(c, agent_id, depth - 1, other_id)
            
            return v / len(children)


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
