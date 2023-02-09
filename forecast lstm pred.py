import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error


# Dataset
data_file = open('01.txt', 'r')
days_with_known_temp = int(data_file.readline())
train_max_temps = [[float(data_file.readline())] for i in range(days_with_known_temp)]
days_with_unknown_temp = int(data_file.readline())
test_max_temps = [[float(data_file.readline())] for i in range(days_with_unknown_temp)]
scaler = MinMaxScaler()
scaler.fit(train_max_temps)
train_data = scaler.transform(train_max_temps)
test_data = scaler.transform(test_max_temps)

# Hyper-parameters
input_size = 1
output_size = 1
num_layers = 1
learning_rate = 10
epochs = 400

# LSTM NN
class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(LSTMNet, self).__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, output_size, num_layers=num_layers, dtype=torch.double)


    def forward(self, x, h0=None, c0=None):
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.num_layers, self.output_size, dtype=torch.double)
            c0 = torch.zeros(self.num_layers, self.output_size, dtype=torch.double)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, (hn, cn)

model = LSTMNet(input_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for i in range(epochs):
    train_input = Variable(torch.tensor(train_data, dtype=torch.double))
    prediction, (hn, cn) = model(train_input)
    target = train_input

    optimizer.zero_grad()
    loss = criterion(prediction, target)
    print(f'Loss {i}: {loss}')
    loss.backward()
    optimizer.step()

# RMSE
test_input = torch.tensor(test_data, dtype=torch.double)
test_prediction, (hn, cn) = model(test_input, hn, cn)
previous_day_fake_predictions = test_max_temps[:-1]
previous_day_fake_predictions.insert(0, train_max_temps[-1])
previous_day_fake_predictions = torch.tensor(previous_day_fake_predictions, dtype=torch.double)

test_input_inverse_transformed = torch.tensor(scaler.inverse_transform(test_data), dtype=torch.double)
test_prediction_inverse_transformed = torch.tensor(scaler.inverse_transform(test_prediction.detach().numpy()), dtype=torch.double)
test_prediction_rmse_loss = np.sqrt(criterion(test_prediction_inverse_transformed, test_input_inverse_transformed).data)
print(f'Total Test Prediction RMSE Loss: {test_prediction_rmse_loss}')
previous_day_fake_rmse_loss = np.sqrt(criterion(previous_day_fake_predictions, test_input_inverse_transformed).data)
print(f'Total Fake Prediction RMSE Loss: {previous_day_fake_rmse_loss}')

# Visualising Results
test_input_inverse_transformed = test_input_inverse_transformed.detach().numpy().squeeze()
test_prediction_inverse_transformed = test_prediction_inverse_transformed.detach().numpy().squeeze()
previous_day_fake_predictions = previous_day_fake_predictions.detach().numpy().squeeze()
test_df = pd.DataFrame({'Test Input': test_input_inverse_transformed,
                        'Test Prediction': test_prediction_inverse_transformed,
                        'Fake Prediction': previous_day_fake_predictions})
print(test_df.head(10))
