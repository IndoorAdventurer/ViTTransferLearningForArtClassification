import torch
from python_package.learning_problems import get_vit_b_32_problem, get_vgg19_problem
from python_package.training import test

model, dl = get_vgg19_problem(False, pretrained=False)
model.load_state_dict(torch.load("../vgg19_ots_fullsize_ds_adam_opt.pth", map_location=torch.device('cpu')))

loss = torch.nn.CrossEntropyLoss()

test(model, dl, loss)