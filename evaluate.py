from game2048.game import Game
from game2048.displays import Display
from RNN_training import RNN


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    # from game2048.agents import RandomAgent as TestAgent
    # from game2048.agents import ExpectiMaxAgent as TestAgent
    # from game2048.agents import LDA as TestAgent
    # from game2048.agents import QDA as TestAgent
    # from game2048.agents import KNN as TestAgent
    from game2048.agents import RNN_A as TestAgent
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)

    print(scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
