from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent,RandomAgent, ExpectiMaxAgent
# 0: left, 1: down, 2: right, 3: up

display1 = Display()
display2 = IPythonDisplay()

# to test the random directions
# game = Game(4, random=False)
# display1.display(game)
# display2.display(game)
# agent = RandomAgent(game,display=display1)
# agent.play(verbose=True)

# to play the game in the shell
# game = Game(4, random=False)
# agent = Agent(game, display=display2)
# agent.play(verbose=True)

#跑到2048
# %%time
# game = Game(4, score_to_win=2048, random=False)
# display2.display(game)
# agent = ExpectiMaxAgent(game,display=display2)
# agent.play(verbose=True)

# %%time
# a random start随机的一个棋盘开始
game = Game(4, score_to_win=512, random=True)
display2.display(game)
agent = ExpectiMaxAgent(game, display=display2)
agent.play(verbose=True)

# 不明白部分
# game = Game(4, score_to_win=2048, random=False)
# # display2.display(game)
# agent = ExpectiMaxAgent(game)
# # agent.play(verbose=True)
# for _ in range(10):
#     print("The ExpectiMax agent always search a fixed solution given certain board:",
#           agent.step())


# 导出该棋盘对应的方向
# print("Running the loop manually...")
#
# game = Game(4, random=False, enable_rewrite_board=False)
# agent = RandomAgent(game)
#
# for _ in range(10):
#     direction = agent.step()
#     print("Moving to direction `%s`..."%direction)
#     game.move(direction)
#     display1.display(game)
#     display2.display(game)

#
# game = Game(4, score_to_win=2048, random=False, enable_rewrite_board=False)
# agent = ExpectiMaxAgent(game)
#
# while game.end == 0:
#     present_board = game.board
#     print("present board is : \n", present_board)
#     direction = agent.step()
#     print("present direction is : ", direction)
#     game.move(direction)



