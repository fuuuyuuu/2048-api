import  pandas as pd
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent,RandomAgent, ExpectiMaxAgent
import numpy as np
# 0: left, 1: down, 2: right, 3: up

game = Game(4, score_to_win=2048, random=False, enable_rewrite_board=False)
agent = ExpectiMaxAgent(game)
while game.end == 0:
    present_board = game.board
    print(type(present_board))
    present_board.reshape(1,16)
    print("present board is : \n", present_board)
    # nb2 = nb.astype(int)
    # nb2[np.where(nb2==0)] = 1
    # nb2 = np.log2(nb2)
    # nb2 = nb2.astype(int)
    # print("present board is : \n", nb2)
    # direction = agent.step()
    # print("present direction is : ",direction
    #       )
    # nb3 = np.array(direction)
    # nb2 = np.append(nb2.tolist(),nb3.tolist())
    # # np.r_[nb2,nb3]
    # print("present direction is : ", nb2)
    # game.move(direction)