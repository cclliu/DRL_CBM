import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


