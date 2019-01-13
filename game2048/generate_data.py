import  pandas as pd
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent,RandomAgent, ExpectiMaxAgent
import numpy as np
import csv
# 0: left, 1: down, 2: right, 3: up

with open("data_random.csv","a") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["r1c1","r1c2","r1c3","r1c4",
                     "r2c1","r2c2","r2c3","r2c4",
                     "r3c1","r3c2","r3c3","r3c4",
                     "r4c1","r4c2","r4c3","r4c4",
                      "dir"])

    for i in range(2000):
        game = Game(4, score_to_win=2048, random=True, enable_rewrite_board=False)
        agent = ExpectiMaxAgent(game)

        while game.end == 0:
            present_board = game.board
            board = present_board.reshape(1,16)
            # board = board.astype(int)
            board = np.int32(board)
            board[np.where(board == 0)] = 1
            board = np.log2(board)
            board = np.int32(board)
            print(",,: ",board)
            # board = board.astype(int)
            direction = agent.step()
            direction_np = np.array(direction)
            direction_np = np.int32(direction_np)
            data = np.append(board.tolist(), direction_np.tolist())
            writer.writerow(data)
            # print("present direction is : ", data)
            game.move(direction)

        i = i+1