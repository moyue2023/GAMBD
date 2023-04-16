import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

        # self.dropout = nn.Dropout(p=0.3)  # dropout训练
        # self.softmax = nn.Softmax()

    def forward(self, x):
        embed_x = self.embed(x)
        self.embed_x = embed_x.detach()
        self.embed_x.requires_grad = True

        # Channel first
        x = torch.transpose(self.embed_x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        # dropout
        # x = self.dropout(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x

    def embedll(self, x):
        embed_x = self.embed(x)
        return embed_x

    def withoutembed(self, embed_x):

        self.embed_x = embed_x.detach()
        self.embed_x.requires_grad = True
        x = torch.transpose(self.embed_x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x
