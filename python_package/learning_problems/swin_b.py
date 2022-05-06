import transformers
from .defaults import dataloaders, freezeLayers
from torch import nn
import transformers

# Ugly monkey-patch because that transformers lib won't output actual predictions like a
# normal person..
transformers.SwinForImageClassification.oldforward = transformers.SwinForImageClassification.forward
def newforward(self, *args, **kwargs):
    return self.oldforward(*args, **kwargs).logits
transformers.SwinForImageClassification.forward = newforward

def get_swin_b_problem(off_the_shelf: bool, dl = dataloaders):
    """
    Returns the whole problem statement for training swin_b on the Rijksdataset.
    In other words: a pre-trained model (with the head replaced), and the dataloaders.\n
    :off_the_shelf: says if it should freeze all but the new head for learning.\n
    :dataloaders: allows user to specify custom dataset.\n
    :pretrained: states if it should load a model pretrained om ImageNet.\n
    """
    model = transformers.SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

    # Prepare for off the shelf learning if needed:
    freezeLayers(model, off_the_shelf)
    
    # Replace head with one that fits the task
    model.classifier = nn.Linear(1024, len(dl.materials))

    return model, dl