import torch
from torch import nn
import pandas as pd
from ..data_loading import RijksDataloaders

def test(model: nn.Module, dataloaders: RijksDataloaders, lossfunc, name: str = "default"):
    """
    ### Tests the given model on dataloaders.test.
    Saves the following data as csv-files:\n
    \t(1) '{name}-test_predictions.csv' gives the full softmax prediction, as well as the correct output;\n
    \t(2) '{name}-test_confusion.csv' gives the confusion matrix.\n
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Gives the softmax predictions of the model, plus the actual output:
    dfPredictions = pd.DataFrame(columns=dataloaders.materials + ["actual_idx", "actual", "batch_loss"])

    # Gives the confusion matrix, with rows showing predictions, and colums what it was supposed to be
    # I realize that this is the wrong way around.. Transposing it for analysis en ploting later on.
    confdims = len(dataloaders.materials)
    confusion = torch.zeros((confdims, confdims), dtype=torch.int)

    with torch.no_grad():
        for x, y in dataloaders.test:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass:
            pred = model(x).softmax(dim=1)
            loss = lossfunc(pred, y)

            # Update statistics:
            for p, a in zip(pred, y):
                newRow = pd.DataFrame(
                    [list(p.cpu().numpy()) + [a.item(), dataloaders.materials[a.item()], loss.item()]],
                    columns=dfPredictions.columns
                )
                dfPredictions = pd.concat([dfPredictions, newRow], ignore_index=True)

                confusion[p.argmax().item()][a.item()] += 1
  
    # Saving predictions
    dfPredictions.to_csv(f"{name}-test_predictions.csv", index=False)

    # Creating dataframe from the confusion matrix and saving it:
    dfConf = pd.DataFrame(confusion.numpy(), columns=dataloaders.materials)
    dfConf["predictions"] = dataloaders.materials
    columns = dfConf.columns.to_list()
    dfConf = dfConf[columns[-1:] + columns[:-1]]
    dfConf.to_csv(f"{name}-test_confusion.csv", index=False)