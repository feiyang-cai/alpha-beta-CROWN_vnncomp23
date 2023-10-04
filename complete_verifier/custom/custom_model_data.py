#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
This file shows how to use customized models and customized dataloaders.

Use the example configuration:
python abcrown.py --config exp_configs/tutorial_examples/custom_model_data_example.yaml
"""

import os
import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
import arguments
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def simple_conv_model(in_channel, out_dim):
    """Simple Convolutional model."""
    model = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, out_dim)
    )
    return model


def two_relu_toy_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 2 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 2),
        nn.ReLU(),
        nn.Linear(2, out_dim)
    )
    # [relu(x+2y)-relu(2x+y)+2, 0*relu(2x-y)+0*relu(-x+y)]
    model[0].weight.data = torch.tensor([[1., 2.], [2., 1.]])
    model[0].bias.data = torch.tensor([0., 0.])
    model[2].weight.data = torch.tensor([[1., -1.], [0., 0.]])
    model[2].bias.data = torch.tensor([2., 0.])
    return model


def simple_box_data(spec):
    """a customized box data: x=[-1, 1], y=[-1, 1]"""
    eps = spec["epsilon"]
    if eps is None:
        eps = 2.
    X = torch.tensor([[0., 0.]]).float()
    labels = torch.tensor([0]).long()
    eps_temp = torch.tensor(eps).reshape(1, -1)
    data_max = torch.tensor(10.).reshape(1, -1)
    data_min = torch.tensor(-10.).reshape(1, -1)
    return X, labels, data_max, data_min, eps_temp


def box_data(dim, low=0., high=1., segments=10, num_classes=10, eps=None):
    """Generate fake datapoints."""
    step = (high - low) / segments
    data_min = torch.linspace(low, high - step, segments).unsqueeze(1).expand(segments, dim)  # Per element lower bounds.
    data_max = torch.linspace(low + step, high, segments).unsqueeze(1).expand(segments, dim)  # Per element upper bounds.
    X = (data_min + data_max) / 2.  # Fake data.
    labels = torch.remainder(torch.arange(0, segments, dtype=torch.int64), num_classes)  # Fake label.
    eps = None  # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    return X, labels, data_max, data_min, eps


def cifar10(spec, use_bounds=False):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    if use_bounds:
        # Option 1: for each example, we return its element-wise lower and upper bounds.
        # If you use this option, set --spec_type ("specifications"->"type" in config) to 'bound'.
        absolute_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        absolute_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        # Be careful with normalization.
        new_eps = torch.reshape(eps / std, (1, -1, 1, 1))
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        # In this case, the epsilon does not matter here.
        ret_eps = None
    else:
        # Option 2: return a single epsilon for all data examples, as well as clipping lower and upper bounds.
        # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        if eps is None:
            raise ValueError('You must specify an epsilon')
        # Rescale epsilon.
        ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps


def simple_cifar10(spec):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    eps = spec["epsilon"]
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps

class Generator(nn.Module):
    def __init__(self, n_noise=4, n_conditions=1, n_channels=1, image_size=32):
        super(Generator, self).__init__()

        self.init_size = image_size // 2 ** 4
        self.l1 = nn.Sequential(nn.Linear(n_noise+n_conditions, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, 2, 0),

            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 0),

            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 0),

            nn.BatchNorm2d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, n_channels, 3, 1, 0),
            #nn.Tanh(),
        )

    def forward(self, x):
        # x [c, z]
        out = self.l1(x)
        #out = self.l1(torch.cat([c,z], dim=1))
        out = out.view(-1, 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, n_conditions=1, n_channels=1, image_size=32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            #block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(n_channels, 16, bn=False), # 16 x size//2 x size//2
            *discriminator_block(16, 32), # 32 x size//4 x size//4
            *discriminator_block(32, 64), # 64 x size//8 x size//8
            *discriminator_block(64, 128), # 128 x size//16 x size//16
            # *discriminator_block(128, 128), # 128 x size//32 x size//32
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        #self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_conditions), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_conditions))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(-1, 128*2*2)
        label = self.aux_layer(out)

        return label

class Gan(nn.Module):
    def __init__(self, n_noise=4, n_conditions=1, n_channels=1, image_size=32):
        super(Gan, self).__init__()

        self.gen = Generator(n_noise=n_noise, n_conditions=n_conditions, n_channels=n_channels, image_size=image_size)
        self.dis = Discriminator(n_conditions=n_conditions, n_channels=n_channels, image_size=image_size)

    def forward(self, x):
        img = self.gen(x)
        label = self.dis(img)
        return label

class DDPG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
    
    def forward(self, d, v):
        # the d and v are normalized to [0, 1]
        x = torch.cat((d, v), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = 0.5*x + 0.5
        x = -1. * F.relu(1+ (-1 *F.relu(x)))+1
        #x = torch.clamp(x, 0, 1)
        return x

class Dynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
    
    def forward(self, d, v, u):
        # the d and v are normalized to [0, 1]
        x = torch.cat((u, d, v), dim=1)
        x = self.fc1(x)
        return x

class SingleStep(nn.Module):
    def __init__(self, d_normalizer=60.0, v_normalizer=30.0, index=0) -> None:
        super().__init__()
        self.gan = Gan()
        self.ddpg = DDPG()
        self.dynamics = Dynamics()
        self.d_normalizer = d_normalizer
        self.v_normalizer = v_normalizer
        #self.denomalizer = torch.tensor([[d_normalizer, 0],
        #                                  [0, v_normalizer]], dtype=torch.float32)
        self.normalizer = [d_normalizer, v_normalizer]
        self.index = index
    
    def forward(self, x):
        d = x[:, 0:1]/self.d_normalizer
        v = x[:, 1:2]/self.v_normalizer
        z = x[:, 2:6]
        x = torch.cat((d, z), dim=1)
        d_predicted = self.gan(x)
        u = self.ddpg(d_predicted, v)
        x = self.dynamics(d, v, u)[:,self.index:self.index+1] * self.normalizer[self.index]
        #x[:, 0:1] = x[:, 0:1] * self.d_normalizer
        #x[:, 1:2] = x[:, 1:2] * self.v_normalizer
        return x

class MultiStep(nn.Module):
    def __init__(self, d_normalizer=60.0, v_normalizer=30.0, index=0, num_steps=1) -> None:
        super().__init__()
        self.gan = Gan()
        self.ddpg = DDPG()
        self.dynamics = Dynamics()
        self.d_normalizer = d_normalizer
        self.v_normalizer = v_normalizer
        #self.denomalizer = torch.tensor([[d_normalizer, 0],
        #                                  [0, v_normalizer]], dtype=torch.float32)
        self.normalizer = [d_normalizer, v_normalizer]
        self.index = index
        self.num_steps = num_steps
    
    def forward(self, x):
        d = x[:, 0:1]/self.d_normalizer
        v = x[:, 1:2]/self.v_normalizer

        for step in range(self.num_steps):
            z = x[:, 2+step*4:6+step*4]
            gan_input = torch.cat((d, z), dim=1)
            d_predicted = self.gan(gan_input)
            u = self.ddpg(d_predicted, v)
            dynamics_output = self.dynamics(d, v, u)
            d = dynamics_output[:,0:1]
            v = torch.relu(dynamics_output[:,1:2])
        x = dynamics_output[:,self.index:self.index+1] * self.normalizer[self.index]
            
        #z = x[:, 2:6]
        #x = torch.cat((d, z), dim=1)
        #d_predicted = self.gan(x)
        #u = self.ddpg(d_predicted, v)
        #x = self.dynamics(d, v, u)[:,self.index:self.index+1] * self.normalizer[self.index]
        #x[:, 0:1] = x[:, 0:1] * self.d_normalizer
        #x[:, 1:2] = x[:, 1:2] * self.v_normalizer
        return x

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        self.cond_bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = nn.BatchNorm2d(out_channels)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.cond_bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out

class cGAN_concat_SAGAN_Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, dim_c=1, g_conv_dim=16):
        super(cGAN_concat_SAGAN_Generator, self).__init__()

        self.z_dim = z_dim
        self.dim_c = dim_c
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=z_dim+dim_c, out_features=g_conv_dim*16*4*4)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.self_attn = Self_Attn(g_conv_dim*16)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.block3 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        #self.block4 = GenBlock(g_conv_dim*4, g_conv_dim*2)
        #self.block5 = GenBlock(g_conv_dim*2, g_conv_dim)
        self.bn = nn.BatchNorm2d(g_conv_dim*16, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim*16, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # n x z_dim
        # x, labels, z
        act0 = self.snlinear0(x)            # n x g_conv_dim*16*1*1
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 1 x 1
        act1 = self.block1(act0)    # n x g_conv_dim*16 x 2 x 2
        act2 = self.block2(act1)    # n x g_conv_dim*8 x 4 x 4
        act2 = self.self_attn(act2)         # n x g_conv_dim*4 x 8 x 8
        act3 = self.block3(act2)    # n x g_conv_dim*4 x 8 x 8
        #act4 = self.block4(act3)    # n x g_conv_dim*2 x 16 x 16
        #act5 = self.block5(act4)    # n x g_conv_dim  x 32 x 32
        act5 = self.bn(act3)                # n x g_conv_dim  x 32 x 32
        act5 = self.relu(act5)              # n x g_conv_dim  x 32 x 32
        act6 = self.snconv2d1(act5)         # n x 3 x 32 x 32
        act6 = self.tanh(act6)              # n x 3 x 32 x 32
        return act6

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)
        out = x + x0
        return out

class cGAN_concat_SAGAN_Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim=16):
        super(cGAN_concat_SAGAN_Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.block1 = DiscBlock(3, d_conv_dim*8)
        self.self_attn = Self_Attn(d_conv_dim*8)
        self.block2 = DiscBlock(d_conv_dim*8, d_conv_dim*8)
        self.block3 = DiscBlock(d_conv_dim*8, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16*8*8, out_features=1)
        self.snlinear2 = snlinear(in_features=d_conv_dim*16*8*8, out_features=1)

    def forward(self, x):
        # n x 3 x 128 x 128
        h1 = self.block1(x)    # n x d_conv_dim*2 x 8 x 8
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 8 x 8
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 4 x 4
        h3 = self.block3(h2, False)    # n x d_conv_dim*8 x  2 x  2
        h4 = self.block4(h3, False)    # n x d_conv_dim*16 x 1 x  1
        #print(h4.shape)
        #h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 1 x 1
        out = self.relu(h4)              # n x d_conv_dim*16 x 1 x 1
        # out = torch.sum(out, dim=[2,3])   # n x d_conv_dim*16
        out = out.view(-1,self.d_conv_dim*16*8*8)
        #output = torch.squeeze(self.snlinear1(out))
        output_1 = torch.sigmoid(self.snlinear2(out)) # n

        return output_1

class GanViT(nn.Module):
    def __init__(self):
        super(GanViT, self).__init__()

        self.gen = cGAN_concat_SAGAN_Generator(z_dim=4, dim_c=1)
        self.dis = cGAN_concat_SAGAN_Discriminator()

    def forward(self, x):
        img = self.gen(x)
        label = self.dis(img)
        return label

class MultiStepViT(nn.Module):
    def __init__(self, d_normalizer=60.0, v_normalizer=30.0, index=0, num_steps=1) -> None:
        super().__init__()
        self.gan = GanViT()
        self.ddpg = DDPG()
        self.dynamics = Dynamics()
        self.d_normalizer = d_normalizer
        self.v_normalizer = v_normalizer
        #self.denomalizer = torch.tensor([[d_normalizer, 0],
        #                                  [0, v_normalizer]], dtype=torch.float32)
        self.normalizer = [d_normalizer, v_normalizer]
        self.index = index
        self.num_steps = num_steps

    
    def forward(self, x):
        d = x[:, 0:1]/self.d_normalizer
        v = x[:, 1:2]/self.v_normalizer

        if self.index == 0 and self.num_steps==1:
            ratio = self.dynamics.fc1.weight.data[0, 2]
            x = d + ratio * v

        else:
            for step in range(self.num_steps):
                z = x[:, 2+step*4:6+step*4]
                gan_input = torch.cat((d, z), dim=1)
                d_predicted = self.gan(gan_input)
                u = self.ddpg(d_predicted, v)
                dynamics_output = self.dynamics(d, v, u)
                d = dynamics_output[:,0:1]
                v = torch.relu(dynamics_output[:,1:2])
            x = dynamics_output[:,self.index:self.index+1] * self.normalizer[self.index]
            
        #z = x[:, 2:6]
        #x = torch.cat((d, z), dim=1)
        #d_predicted = self.gan(x)
        #u = self.ddpg(d_predicted, v)
        #x = self.dynamics(d, v, u)[:,self.index:self.index+1] * self.normalizer[self.index]
        #x[:, 0:1] = x[:, 0:1] * self.d_normalizer
        #x[:, 1:2] = x[:, 1:2] * self.v_normalizer
        return x

class MultiStepTaxiNet(nn.Module):
    def __init__(self, index=0, num_steps=1, p_normalizer=6.366468343804353, theta_normalizer=17.248858791583547) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),

            nn.Linear(8, 2),

            nn.Linear(2, 1, bias=False)
        )
        
        self.normalizer = [p_normalizer, theta_normalizer]
        self.index = index
        self.num_steps = num_steps
    
    def dynamics(self, p, theta, u):

        for i in range(20):
            p = p + 5*0.05*torch.sin(theta*torch.pi/180.)
            theta = theta + 0.05*180.*(torch.tan(u*torch.pi/180.))/torch.pi
        return torch.cat((p, theta), dim=1)
    
    def forward(self, x):
        p = x[:, 0:1]
        theta = x[:, 1:2]

        for step in range(self.num_steps):
            z = x[:, 2+step*2:4+step*2]
            input = torch.cat((z, p/self.normalizer[0], theta/self.normalizer[1]), dim=1)
            u = self.main(input)
            dynamics_output = self.dynamics(p, theta, u)
            p = dynamics_output[:,0:1]
            theta = dynamics_output[:,1:2]

        x = dynamics_output[:,self.index:self.index+1]

        return x
