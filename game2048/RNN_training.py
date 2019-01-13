import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_loader import data_load
from torch.autograd import Variable
import numpy as np


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 6400
TIME_STEP = 4          # rnn time step / image height
INPUT_SIZE = 4         # rnn input size / image width
global LR;
LR = 0.001               # learning rate\



def DataLoad():
    # board_data loading with a batche size
    train_data = data_load(data_root = 'Train.csv', data_tensor = transforms.Compose([transforms.ToTensor()]))

    X_train = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # test_data = data_load(data_root='Test.csv', data_tensor=transforms.Compose([transforms.ToTensor()]))
    # X_test = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return X_train


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.my_rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=512,
            num_layers=4,
            batch_first=True
        )

        self.out = nn.Linear(512, 4)

    def forward(self, x):

        r_out, (h_n, h_c) = self.my_rnn(x,None)
        out = self.out(r_out[:, -1 ,:])
        return out

def main():
    global LR;
    rnn_training = RNN()
    train_data = DataLoad()                                         
    optimizer = torch.optim.Adam(rnn_training.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        if epoch == 10:
            LR = 0.0001
            optimizer = torch.optim.Adam(rnn_training.parameters(), lr=LR)
        for step, (train, target) in enumerate(train_data):
            target = target.long();
            b_x = Variable(train.view(-1,4,4))
            # print(b_x.shape)
            b_y = Variable(target)

            if torch.cuda.is_available():
                b_x = Variable(b_x).cuda() 
                b_y = b_y.cuda()
                rnn_training = rnn_training.cuda()            

            optimizer.zero_grad()

            output = rnn_training(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()


            if step % 50 == 0:
                train_output = output  # (samples, time_step, input_size)
                # pred_y = torch.max(train_output, 1)[1].data
                pred_y = train_output.data.max(1)[1]
                # print(type(pred_y), type(target))

                num = (pred_y.eq(b_y.data).sum())
                accuracy = 100*num / 6400
                print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)
        torch.save(rnn_training,'rnn_model_b'+str(epoch)+'.pkl')
    torch.save(rnn_training, 'rnn_model_final.pkl')

if __name__ == '__main__':
    main()
