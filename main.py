import random
import time

from WarehouseEnv import WarehouseEnv
import argparse
import submission
import Agent


def run_agents():
    parser = argparse.ArgumentParser(description='Test your submission by pitting agents against each other.')
    parser.add_argument('agent0', type=str,
                        help='First agent')
    parser.add_argument('agent1', type=str,
                        help='Second agent')
    parser.add_argument('-t', '--time_limit', type=float, nargs='?', help='Time limit for each turn in seconds', default=1)
    parser.add_argument('-s', '--seed', nargs='?', type=int, help='Seed to be used for generating the game',
                        default=random.randint(0, 255))
    parser.add_argument('-c', '--count_steps', nargs='?', type=int, help='Number of steps each robot gets before game is over',
                        default=4761)
    parser.add_argument('--console_print', action='store_true')

    parser.add_argument('--screen_print', action='store_true')

    parser.add_argument('--tournament', action='store_true')

    args = parser.parse_args()

    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "greedyImproved": submission.AgentGreedyImproved(),
        "minimax": submission.AgentMinimax(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax(),
        "hardcoded": submission.AgentHardCoded(),
    }

    # agent_names = sys.argv
    agent_names = [args.agent0, args.agent1]
    env = WarehouseEnv()

    if not args.tournament:
        env.generate(args.seed, 2*args.count_steps)

        if args.console_print:
            print('initial board:')
            env.print()

        if args.screen_print:
            env.pygame_print()

        for _ in range(args.count_steps):
            for i, agent_name in enumerate(agent_names):
                agent = agents[agent_name]
                start = time.time()
                op = agent.run_step(env, i, args.time_limit)
                end = time.time()
                if end - start > args.time_limit:
                    raise RuntimeError("Agent used too much time!")
                env.apply_operator(i, op)
                if args.console_print:
                    print('robot ' + str(i) + ' chose ' + op)
                    env.print()
                if args.screen_print:
                    env.pygame_print()
            if env.done():
                break
        balances = env.get_balances()
        print(balances)
        if balances[0] == balances[1]:
            print('draw')
        else:
            print('robot', balances.index(max(balances)), 'wins!')
    else:
        robot0_wins = 0
        robot1_wins = 0
        draws = 0
        num_of_games = 100

        for i in range(num_of_games):
            env.generate(args.seed + i, 2*args.count_steps)
            if args.console_print:
                print('initial board:')
                env.print()
            if args.screen_print:
                env.pygame_print()

            for _ in range(args.count_steps):
                for i, agent_name in enumerate(agent_names):
                    agent = agents[agent_name]
                    start = time.time()
                    op = agent.run_step(env, i, args.time_limit)
                    end = time.time()
                    if end - start > args.time_limit:
                        raise RuntimeError("Agent used too much time!")
                    env.apply_operator(i, op)
                if args.console_print:
                    print('robot ' + str(i) + ' chose ' + op)
                    env.print()
                if args.screen_print:
                    env.pygame_print()
                if env.done():
                    break
            balances = env.get_balances()
            if balances[0] == balances[1]:
                draws += 1
            elif balances[0] > balances[1]:
                robot0_wins += 1
            else:
                robot1_wins += 1
        print("Robot 0 wins: ", robot0_wins)
        print("Robot 1 wins: ", robot1_wins)
        print("Draws: ", draws)




if __name__ == "__main__":
    run_agents()
