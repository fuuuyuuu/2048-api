import torch
import numpy as np
import pandas as pd

class data_load(torch.utils.data.Dataset):
    def __init__(self, data_root, data_tensor=None, target_tensor=None):
        csv_data = pd.read_csv(data_root)
        csv_data = csv_data.values
        self.board_data = csv_data[:, 0:16]
        print("board_size:   ", np.shape(self.board_data))
        self.direction_data = csv_data[:, 16]
        print("direction_size:   ", np.shape(self.direction_data))
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        board = self.board_data[index]
        board = board.reshape((4, 4))

        board = board / 11.0
        board = board[:, :, np.newaxis]
        direction = self.direction_data[index]
        direction = direction.astype(np.int32)

        if self.data_tensor is not None:
            board = self.data_tensor(board)
            board = board.float()
            # print("test_pp:  ", type(board))

        return board, direction

    def __len__(self):
        return len(self.direction_data)