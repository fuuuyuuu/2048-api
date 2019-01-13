import numpy as np
from sklearn.externals import joblib
import torch
import torchvision
import time



class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class LDA(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)


        clf2 = joblib.load('LDA_model.pkl')
        self.search_func = clf2.predict
        # print(clf2.predict(board))

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 16)
        board = np.int8(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = np.int8(board)
        board = board / 11.0
        direction = self.search_func(board)
        return int(direction)

class QDA(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)


        clf3 = joblib.load('QDA_model.pkl')
        self.search_func = clf3.predict

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 16)
        board = np.int8(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = np.int8(board)
        board = board / 11.0
        direction = self.search_func(board)
        return int(direction)


class KNN(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)


        clf4 = joblib.load('KNN_3_model.pkl')
        self.search_func = clf4.predict

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 16)
        board = np.int32(board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = np.int32(board)
        board = board / 11.0
        direction = self.search_func(board)
        return int(direction)


class RNN_A(Agent):
    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        # if(self.game.score < 512):
        #     clf1 = torch.load('rnn_model_4.pkl', map_location='cpu')
        #     print("--------present MODEL is rnn_model_4---------")
        # # else:
        # #     clf4 = torch.load('rnn_model_8.pkl', map_location='cpu')
        # #     print("--------present MODEL is rnn_model_8---------")
        # self.search_func = clf1

    def step(self):
        time0 = time.time()
        present_board = self.game.board
        board = np.int32(present_board)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        # board = np.int32(board)
        board1 = np.rot90(board)
        # print(board1)
        board2 = np.rot90(board1)
        board3 = np.rot90(board2)
        #正常方向
        board = board / 11.0
        board = board[:, :, np.newaxis]

        board = torchvision.transforms.ToTensor()(board)
        board = board.float()

        list1 = [0,0,0,0]
        clf3 = torch.load('rnn_model_b15.pkl', map_location='cpu')
        self.search_func = clf3
        direction3 = self.search_func(board)
        direction3 = direction3.data.max(1)[1]
        list1[int(direction3)] += 1

        # board = torch.unsqueeze(board, dim=0).float()

        #转90
        board1 = board1 / 11.0
        board1 = board1[:, :, np.newaxis]
        
        board1 = torchvision.transforms.ToTensor()(board1)
        board1 = board1.float()
        direction3 = self.search_func(board1)
        direction3 = direction3.data.max(1)[1]
        direction3 = (direction3 + 3)%4
        # print(direction3)
        list1[int(direction3)] += 1
        
        #转180
        board2 = board2 / 11.0
        board2 = board2[:, :, np.newaxis]
        
        board2 = torchvision.transforms.ToTensor()(board2)
        board2 = board2.float()
        direction3 = self.search_func(board2)
        direction3 = direction3.data.max(1)[1]
        direction3 = (direction3 + 2) % 4
        list1[int(direction3)] += 1
        
        #转270
        board3 = board3 / 11.0
        board3 = board3[:, :, np.newaxis]
        
        board3 = torchvision.transforms.ToTensor()(board3)
        board3 = board3.float()
        direction3 = self.search_func(board3)
        direction3 = direction3.data.max(1)[1]
        direction3 = (direction3 + 1) % 4
        list1[int(direction3)] += 1




        if (self.game.score < 512):
            clf1 = torch.load('rnn_model_b8.pkl', map_location='cpu')
            self.search_func = clf1
            direction1 = self.search_func(board)
            direction1 = direction1.data.max(1)[1]
            list1[int(direction1)] += 1

            direction3 = self.search_func(board1)
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 3)%4
            # print(direction3)
            list1[int(direction3)] += 1

            direction3 = self.search_func(board2)        
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 2) % 4
            list1[int(direction3)] += 1

            direction3 = self.search_func(board3)
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 1) % 4
            list1[int(direction3)] += 1


        else:
            clf4 = torch.load('rnn_model_b18.pkl', map_location='cpu')
            self.search_func = clf4
            direction4 = self.search_func(board)
            direction4 = direction4.data.max(1)[1]
            list1[int(direction4)] += 1

            direction3 = self.search_func(board1)
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 3)%4
            # print(direction3)
            list1[int(direction3)] += 1

            direction3 = self.search_func(board2)        
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 2) % 4
            list1[int(direction3)] += 1

            direction3 = self.search_func(board3)
            direction3 = direction3.data.max(1)[1]
            direction3 = (direction3 + 1) % 4
            list1[int(direction3)] += 1

        direction = list1.index(max(list1))
        t1 = time.time() - time0
        print(t1)



        # direction = self.search_func(board)
        # direction = direction.data.max(1)[1]
        # if(self.game.score >= 256):
        #     clf4 = torch.load('rnn_model_18.pkl', map_location='cpu')
        #     self.search_func = clf4
        #     print("--------present MODEL is rnn_model_18---------")
        return int(direction)