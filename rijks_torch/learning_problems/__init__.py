# A bunch of functions stating learning problems:

#    CNN based:
from .vgg import get_vgg19_problem
from .vgg import get_vgg19_drop_problem
from .resnet import get_resnet50_problem
from .resnet import get_resnet50_drop_problem
from .convnext import get_convnext_b_problem
from .convnext import get_convnext_s_problem
from .convnext import get_convnext_t_problem
from .efficientnetv2 import get_efficientnetv2_m_problem
from .efficientnetv2 import get_efficientnetv2_t_problem

#    ViT based:
from .vit import get_vit_b_16_problem
from .vit import get_vit_t_16_problem
from .swin import get_swin_b_problem
from .swin import get_swin_s_problem
from .swin import get_swin_t_problem
from .beit import get_beit_b_16_problem
from .beit import get_beit_b_16_drop_problem
from .deit import get_deit_b_16_problem
from .deit import get_deit_t_16_problem