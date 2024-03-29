from .defaults import freezeLayers
from torch import nn
import timm

def get_efficientnetv2_m_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training efficientnetv2_m on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('efficientnetv2_rw_m', pretrained=pretrained)
    
    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.classifier = nn.Linear(2152, len(dl.materials))

    return model, dl


def get_efficientnetv2_m_drop_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """ Same but with a dropout layer. This version is used for fine tuning """
    model, dl = get_efficientnetv2_m_problem(off_the_shelf, dl, pretrained)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(2152, len(dl.materials))
    )
    return model, dl


def get_efficientnetv2_t_problem(off_the_shelf: bool, dl, pretrained: bool = True):
    """
    Returns the whole problem statement for training efficientnetv2_t on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = timm.create_model('efficientnetv2_rw_t', pretrained=pretrained)
    
    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1024, len(dl.materials))
    )

    return model, dl