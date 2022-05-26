# Contains defaul values that most models will likely use
# After update it only contains methods to create defaults

from torchvision import transforms


def freezeLayers(model, off_the_shelf: bool):
    """Prepare model for off the shelf learning if off_the_shelf==True"""
    if off_the_shelf:
        for param in model.parameters():
            param.requires_grad = False

def buildTransform(imnet_norm: bool, imsize: int = 224, extratransforms = None) -> transforms.Compose:
    """
    Builds a transform that modifies input images in a dataset.\n
    imnet_norm: if true, transforms.Normalize images with mean and std of ImageNet.\n
    imsize: required size for input images.\n
    extratransforms gets appended to the returned transforms.Compose object. Note that
    these get applied last! (though for rotation it does not seem to matter that there
    are black patches around).\n
    """
    if extratransforms != None and not isinstance(extratransforms, list):
        extratransforms = [extratransforms]
    
    tfs = [
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize)
    ]

    if imnet_norm:
        tfs += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    
    if extratransforms != None:
        # NOTE: these get applied last, so something like a rotation will show plack around it
        tfs += extratransforms
    
    return transforms.Compose(tfs)