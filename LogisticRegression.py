import torch
from torch import tensor
from torch import sigmoid
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.0], [0.0], [1.0], [1.0]])

# first step designing the model using class and variables
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1) # for one input and one output
    
    def forward(self, x):
        # here the forward function accepts a variable of input data and we must return 
        # a variale of output data
        y_pred = sigmoid(self.linear(x))
        return y_pred
model = Model()
print(model)

#step 2 construction of loss function and optimizer. 
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

#step 3 - Training loop - forward, backward and update

for epoch in range(1000):
    # forward pass: compute predicted y by passing x to the model
    y_pred = model(x_data)
    print(y_pred)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
