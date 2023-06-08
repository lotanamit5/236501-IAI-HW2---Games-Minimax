from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


class Agent:
    # returns the next operator to be applied - i.e. takes one turn
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()

    # returns list of legal operators and matching list of states reached by applying them
    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)
        return robot.credit - other_robot.credit


# picks random operators from the legal ones
class AgentRandom(Agent):
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)
        return random.choice(operators)


class AgentGreedy(Agent):
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        children_heuristics = [self.heuristic(child, robot_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


