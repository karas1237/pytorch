import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)    

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]] # 5 X 3
y_data = [[152.], [185.], [180.], [196.], [142.]] # 5 X 1

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

print(X)
# Our hypothesis XW+b
model = nn.Linear(3, 1, bias=True)

# cost criterion
criterion = nn.MSELoss()

# Minimize
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model
for step in range(2001):
    optimizer.zero_grad()

    hypothesis = model(X)

    cost = criterion(hypothesis, Y)

    cost.backward()
    optimizer.step()

    if step % 10 == 0:
        # print(hypothesis)
        print(step, cost.data.numpy(), model.weight.data.numpy(), model.bias.data.numpy())

