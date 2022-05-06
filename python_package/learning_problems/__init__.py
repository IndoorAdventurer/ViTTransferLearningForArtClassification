# A bunch of functions stating learning problems:
#    CNN based:
from .vgg19 import get_vgg19_problem
from .convnext_b import get_convnext_b_problem
from .resnet50 import get_resnet50_problem

#    ViT based:
from .vit_b_32 import get_vit_b_32_problem
from .swin_b import get_swin_b_problem
