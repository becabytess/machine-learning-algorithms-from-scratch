import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("multipliers.csv")

# Preprocess the data
sequence_length = 20
data = data["Multiplier"].values
data = data.reshape(-1, 1)  # Reshape to a single column

# Normalize the data
data = data / np.max(data)

# Create sequences and labels
sequences = []
labels = []
for i in range(len(data) - sequence_length):
    sequences.append(data[i : i + sequence_length])
    labels.append(data[i + sequence_length])

# Convert to PyTorch tensors
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
input_size = 1
hidden_size = 50
model = LSTMModel(input_size, hidden_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(sequences)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to predict next number given a sequence
def predict_next(sequence):
    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    prediction = model(sequence)
    return prediction.item() * np.max(data)  # Rescale the prediction

# Test
test_sequence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
predicted_next = predict_next(test_sequence)
print("Predicted next number:", predicted_next)
