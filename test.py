import rijks_torch.learning_problems as probs
import rijks_torch.learning_problems.defaults as defs
from   rijks_torch.training import train, test

from torch import nn, optim

# Custom dataloader:
customDl = defs.RijksDataloaders(
    ds_name=defs.dataset_fullsize,
    hist_path=defs.hist_path,
    img_dir=defs.img_dir,
    transforms={"all": defs.buildTransform(imnet_norm=True)},
    batch_size=32
)

# Getting the whole problem:
model, dl = probs.get_vgg19_problem(True, customDl)

loss = nn.CrossEntropyLoss()
optimizer = optim.RMSprop([param for param in model.parameters() if param.requires_grad == True], lr=0.001)
#optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True])

# Training (and validating) the model:
model = train(model, dl, loss, optimizer, 100)

# Testing the model:
test(model, dl, loss)