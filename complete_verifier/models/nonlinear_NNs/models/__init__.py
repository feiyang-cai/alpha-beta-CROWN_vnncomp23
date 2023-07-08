from models.vit import *
from models.siren import *

Models = {
    'vit_1_mnist': ViT_1layer_mnist,
    'vit_1_cifar': ViT_1layer_cifar,
    'vit_2_cifar': ViT_2layer_cifar,
    'vit_2_mnist': ViT_2layer_mnist,
    'siren': Siren,
}
