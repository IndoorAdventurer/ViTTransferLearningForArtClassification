import sys
from torch import nn, optim
import torch
import rijks_torch.learning_problems as probs
import rijks_torch.learning_problems.defaults as defs
from rijks_torch.data_loading.rijksdataloaders import RijksDataloaders
from rijks_torch.training import train, test


def main():
    """
    Arguments given to script must be:\n
    1) name of the experiment: will be prepended to all output files;\n
    2) name of the model such that 'get_{name}_problem()' is defined;\n
    3) dataset files. e.g. if /x/y/z is given, then
        /x/y/z-train.csv,
        /x/y/z-val.csv,
        /x/y/z-test.csv,
        /x/y/z-hist.csv,must exist;\n
    4) name of directory containing all the jpg files (should be stored locally, so under /local)
    """
    
    # If there is no gpu, we aren't actually going to run it..
    assert torch.cuda.is_available(), "There was no GPU :-("

    experiment_name = sys.argv[1]
    model_name = sys.argv[2]
    dataset_files = sys.argv[3]
    dataset_jpg_dir = sys.argv[4]

    # Creating the dataloaders from given arguments:
    datloader = RijksDataloaders(
        ds_name=dataset_files,
        hist_path=dataset_files + "-hist.csv",
        img_dir=dataset_jpg_dir,
        transforms={"all": defs.buildTransform(imnet_norm=True)},
        batch_size=64
    )

    # Get the model tailored to specification. Using getattr because function from cli args
    model, dl = getattr(probs, f"get_{model_name}_problem")(off_the_shelf=False, dl=datloader)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=5e-6)

    # Training and validating (best model on val set returned):
    model = train(model, dl, loss, optimizer, name=experiment_name)

    # Testing model that performed best on validation set:
    test(model, dl, loss, name=experiment_name)

if __name__ == "__main__":
    main()