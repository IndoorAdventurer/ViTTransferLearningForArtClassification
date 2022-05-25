import matplotlib.pyplot as plt
from IPython import display
from numpy import disp
import pandas as pd

def updatePlots(csv_file: str):
    """"Show plots interactively: update after each validation run"""
    display.clear_output()
    df = pd.read_csv(csv_file)

    epochs = [x + 1 for x in range(len(df))]
    loss = df["mean_loss"].to_list()
    acc = df["accuracy"].to_list()

    # Give a horizontal line if only one datapoint available:
    if len(epochs) == 1:
        epochs += [1.1]
        loss += loss
        acc += acc

    plt.plot(epochs, loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    plt.plot(epochs, acc)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()