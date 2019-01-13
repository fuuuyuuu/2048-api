# import torch
# from torch.autograd import variable
# import numpy as np
# import pandas as pd
#
# # data loading...
# csv_data = pd.read_csv('Train.csv')
# csv_data = csv_data.values
# board_data = csv_data[:,0:16]
# # print(board_data[0:4,:])
# direction_data = csv_data[:,16]
# # print(type(board_data))
# X = np.int32(board_data)/11.0
# Y = np.int32(direction_data)
# X_tensor = torch.from_numpy(X)
# Y_tensor = torch.from_numpy(Y)

class