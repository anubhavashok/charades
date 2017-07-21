import torch
from torch import nn
from torch.autograd import Variable
import config

class LSTMTransformer(nn.Module):
    def __init__(self, input_size=config.HIDDEN_SIZE, hidden_size=config.HIDDEN_SIZE):
        super(LSTMTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, 2, bidirectional=False)
    
    def forward(self, input):
        # Define hidden state
        h = (Variable(torch.zeros(2, 1, self.hidden_size)).cuda(),
             Variable(torch.zeros(2, 1, self.hidden_size)).cuda())
        input = input.unsqueeze(1)
        output, _ = self.lstm(input, h)
        return output.squeeze(1)
