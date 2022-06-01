import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from ..data_loading import RijksDataloaders

from time import time
from copy import deepcopy

def train(model: nn.Module,
          dataloaders: RijksDataloaders,
          lossfunc, optimizer, max_epochs:
          int = 200,
          early_stop: int = 7,
          scheduler: optim.lr_scheduler.ReduceLROnPlateau = None,
          name: str = "default"):
    """
    ## Function for training of a model

    model should be all set, and adapted to the specified datasets in terms of architecture.\n
    dataloaders is a class encapsulating the training, validating and testing sets.\n
    lossfunc is the used loss function.\n
    optimizer is the used optimizer.\n
    max_epochs maximum number of epochs if no early stopping occured\n
    early_stop if no new best is found after this many trials, training stops\n
    scheduler dynamically changes the learning rate\n
    name is the name of the experiment. Gets prepended to output filenames\n
    ### Returns the model that scored best on the validation set!\n
    ### Also saves validation statistics to {name}-validation.csv
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    startTime = time()

    # Keeping track of the best model:
    best_model = deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0

    # File to record validation statistics to:
    validation_log = f"{name}-validation.csv"

    # Preparing the validation csv file:
    with open(validation_log, "a") as f:
        f.write("accuracy,mean_loss\n")

    for epoch in range(max_epochs):
        train_loop(model, dataloaders.train, lossfunc, optimizer, device)
        loss = validation_loop(model, dataloaders.val, lossfunc, device, validation_log)
        
        # Scheduler will always be ReduceLROnPlateau, so needs loss as input
        if scheduler != None:
            scheduler.step(loss)

        # Save best model and early stop if no new best for too long:
        if loss < best_loss:
            best_model = deepcopy(model.state_dict())
            best_loss = loss
            best_epoch = epoch
            torch.save(best_model, f"{name}-best.pth")
        elif epoch - best_epoch >= early_stop:
            break

    endTime = time()
    with open(f"{name}-trainingtime.txt", "w") as f:
        f.write(f"Training time: {endTime - startTime}")

    # Return the best model found:
    model.load_state_dict(best_model)
    return model

def train_loop(model: nn.Module, dataloader: DataLoader, lossfunc, optimizer, device):
    model.train()

    # Scalar for mixed precision computations:
    scalar = GradScaler()

    for batchnum, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass:
        with autocast(): # Taking advantage of mixed precision speedup (on v100)
            pred = model(x)
            loss = lossfunc(pred, y)

        # Backward pass:
        optimizer.zero_grad(set_to_none=True)
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

def validation_loop(model: nn.Module, dataloader: DataLoader, lossfunc, device, validation_csv):
    model.eval()

    running_accuracy = 0.0
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass:
            pred = model(x)
            loss = lossfunc(pred, y)

            # Update statistics:
            for p, a in zip(pred, y):
                count += 1
                running_accuracy += (1 / count) * ((p.argmax() == a).type(torch.float).item() - running_accuracy)
                running_loss += (1 / count) * (loss.item() - running_loss)

    # Save statistics to validation file:
    with open(validation_csv, "a") as f:
        f.write(f"{running_accuracy},{running_loss}\n")

    return running_loss