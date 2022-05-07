import python_package.learning_problems as probs
from python_package.training import train

from torch import nn, optim

model, dl = probs.get_beit_b_16_problem(True)

loss = nn.CrossEntropyLoss()
optimizer = optim.RMSprop([param for param in model.parameters() if param.requires_grad == True])

train(model, dl, loss, optimizer, 5)