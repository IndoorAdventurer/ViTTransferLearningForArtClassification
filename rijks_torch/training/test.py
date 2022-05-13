import torch
from torch import nn
import pandas as pd
from ..data_loading import RijksDataloaders

def test(model: nn.Module, dataloaders: RijksDataloaders, lossfunc):
    """
    ### Tests the given model on dataloaders.test.
    Saves the following data as csv-files:\n
    \t(1) 'test_predictions.csv' gives the full softmax prediction, as well as the correct output;\n
    \t(2) 'test_confusion.csv' gives the confusion matrix.\n
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    print("TESTING HAS STARTED!")

    # Gives the softmax predictions of the model, plus the actual output:
    dfPredictions = pd.DataFrame(columns=dataloaders.materials + ["actual_idx", "actual", "batch_loss"])

    # Gives the confusion matrix, with rows showing predictions, and colums what it was supposed to be
    confdims = len(dataloaders.materials)
    confusion = torch.zeros((confdims, confdims), dtype=torch.int)

    # Statistics to output at the end
    running_accuracy = 0.0
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataloaders.test:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass:
            pred = model(x).softmax(dim=1)
            loss = lossfunc(pred, y)

            # Update statistics (more precise than for training: taking size of last batch into account):
            for p, a in zip(pred, y):
                newRow = pd.DataFrame(
                    [list(p.cpu().numpy()) + [a.item(), dataloaders.materials[a.item()], loss.item()]],
                    columns=dfPredictions.columns
                )
                dfPredictions = pd.concat([dfPredictions, newRow], ignore_index=True)

                confusion[p.argmax().item()][a.item()] += 1

                # Updating statistics to show at the end:
                count += 1
                running_accuracy += (1 / count) * ((p.argmax() == a).type(torch.float).item() - running_accuracy)
                running_loss += (1 / count) * (loss.item() - running_loss)

            print(f"\033[F\033[KAccuracy: {running_accuracy:0.3f}; Mean loss: {running_loss:0.3f}")
    
    # Saving predictions
    dfPredictions.to_csv("test_predictions.csv", index=False)

    # Creating dataframe from the confusion matrix and saving it:
    dfConf = pd.DataFrame(confusion.numpy(), columns=dataloaders.materials)
    dfConf["predictions"] = dataloaders.materials
    columns = dfConf.columns.to_list()
    dfConf = dfConf[columns[-1:] + columns[:-1]]
    dfConf.to_csv("test_confusion.csv", index=False)

    print("TESTING HAS ENDED!\n"
          "   Results saved to 'test_predictions.csv' and 'test_confusion.csv'\n"
         f"   Final accuracy was: {running_accuracy:0.3f}; Mean loss was: {running_loss:0.3f}")