import torch
from torch import nn
from torch.utils.data import DataLoader
from ..data_loading import RijksDataloaders

from time import time
from copy import deepcopy

def train(model: nn.Module, dataloaders: RijksDataloaders, lossfunc, optimizer, num_epochs: int):
    """
    ## Function for training of a model

    model should be all set, and adapted to the specified datasets in terms of architecture.\n
    dataloaders is a class encapsulating the training, validating and testing sets.\n
    lossfunc is the used loss function.\n
    optimizer is the used optimizer.\n
    num_epochs ... ah you get it.. ;-)\n\n
    ### Returns the model that scored best on the validation set!\n
    ### Also saves statistics of first epoch to first_epoch.csv and validation statistics to validation.csv
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"___TRAINING_STARTED_({device.upper()})___" + "_" * 40)
    startTime = time()

    # Keeping track of the best model
    best_model = deepcopy(model.state_dict())
    best_accuracy = 0.0

    # Record statistics of first batch to this file:
    first_epoch_log = "first_epoch.csv"

    # Preparing the validation csv file:
    with open("validation.csv", "a") as f:
        f.write("accuracy, mean_loss\n")

    for epoch in range(num_epochs):
        print(f"---EPOCH-{(epoch + 1):03}-OF-{num_epochs:03}-" + "-" * 30)

        train_loop(model, dataloaders.train, lossfunc, optimizer, device, first_epoch_log)
        accuracy = validation_loop(model, dataloaders.val, lossfunc, device)

        # Set file to None so subsequent epochs won't be recorded:
        # Reason behind this is that these epochs have seen all instances before
        # so the statistics are less meaningfull.
        first_epoch_log = None

        if accuracy > best_accuracy:
            print("New best model! Saving parameters.")
            best_model = deepcopy(model.state_dict())
            best_accuracy = accuracy
            torch.save(best_model, "best.pth")

    endTime = time()
    print("___TRAINING_ENDED_____" + "_" * 40)
    print(f"Total training time was: {endTime - startTime} seconds.")
    print(f"The best model had an accuracy of: {best_accuracy:0.3f}.")

def train_loop(model: nn.Module, dataloader: DataLoader, lossfunc, optimizer, device, savefile):
    model.train()
    percentDone = -1
    num_batches = len(dataloader)

    running_accuracy = 0.0
    running_loss = 0.0

    # Adding column names to csv file to save to:
    if savefile != None:
        with open(savefile, "a") as f:
            f.write("batch_size, accuracy, mean_loss\n")

    for batchnum, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        batch_size = len(y)

        # Forward pass:
        pred = model(x)
        loss = lossfunc(pred, y)

        # Backward pass:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics (neglecting fact that last batch is smaller):
        batch_acc = (pred.argmax(1) == y).type(torch.float).sum().item() / batch_size
        running_accuracy += (1 / (batchnum + 1)) * (batch_acc - running_accuracy)
        running_loss += (1 / (batchnum + 1)) * (loss.item() - running_loss)

        # Saving statistics to file if filename given:
        if savefile != None:
            with open(savefile, "a") as f:
                f.write(f"{batch_size}, {batch_acc}, {running_loss}\n")

        # Update display 100 times per epoch:
        tmpPercent = int(100 * batchnum / num_batches)
        if tmpPercent > percentDone:
            percentDone = tmpPercent
            showProgress(percentDone, running_loss, running_accuracy)

def showProgress(percentDone: int, running_loss: float, running_acc: float):
    """Just showing off :-p Having a nice function to show progress"""
    total_bars = 35
    num_bars = int(total_bars * percentDone / 100)
    
    # Moving up two lines if it isn't the first time:
    if (percentDone != 0):
        print("\033[2F", end="")
    
    # Clear line:
    print("\033[2K", end="")

    # Printing a loading bar:
    print("(", end="")
    for i in range(num_bars):
        print("=", end="")
    for i in range(total_bars - num_bars):
        print(" ", end="")
    print(f") {percentDone}% done!")

    # Clear line:
    print("\033[2K", end="")

    # Printing loss and accuracy, and going back to beginning of previous line:
    print(f"Training loss: {running_loss:0.3f}; Training accuracy: {running_acc:0.3f}.", flush=True)

def validation_loop(model: nn.Module, dataloader: DataLoader, lossfunc, device):
    model.eval()

    running_accuracy = 0.0
    running_loss = 0.0
    # TODO: Add more statistics :-)

    count = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass:
            pred = model(x)
            loss = lossfunc(pred, y)

            # Update statistics (more precise than for training: taking size of last batch into account):
            for p, a in zip(pred, y):
                count += 1
                running_accuracy += (1 / count) * ((p.argmax() == a).type(torch.float).item() - running_accuracy)
                running_loss += (1 / count) * (loss.item() - running_loss)

    # Print statistics:
    print(
f"""Validation statistics:
    Accuracy: {running_accuracy:0.3f}
    Loss:     {running_loss:0.3f}
"""
    )

    # Save statistics to validation file:
    with open("validation.csv", "a") as f:
        f.write("{running_accuracy}, {running_loss}\n")

    return running_accuracy

