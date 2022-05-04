from python_package.learning_problems import vgg19
from python_package.training import train

from torch import nn, optim

model, dl = vgg19(True)

loss = nn.CrossEntropyLoss()
optimizer = optim.RMSprop([param for param in model.parameters() if param.requires_grad == True])

train(model, dl, loss, optimizer, 5)