# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
import numpy as np


channels = 3

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)
    
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            self.conv1,
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv2,
            nn.InstanceNorm2d(out_channels, affine=True),
            )
        self.conv_sc = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv_sc.weight.data, 1.)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                self.conv_sc
                )
        self.activate = nn.ReLU()
    def forward(self, x):
        return self.activate(self.model(x) + self.bypass(x))


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=64
DISC_SIZE=64

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE*16)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        conv_intermediate = nn.Conv2d(GEN_SIZE*4 + c_dim, GEN_SIZE*4, 3, 1, padding=1)
        nn.init.xavier_uniform(conv_intermediate.weight.data, 1.)
        self.model1 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*16, stride=2),
            ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*8, stride=2),
            ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, stride=2),
            nn.BatchNorm2d(GEN_SIZE*4),
            nn.ReLU()
            )
        self.model2 = nn.Sequential(
            conv_intermediate,
            ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, stride=2),
            ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z, c):
        h = self.model1(self.dense(z).view(-1, GEN_SIZE*16, 4, 4))
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), h.size(2), h.size(3))
        hc = torch.cat([h, c], dim=1)
        return self.model2(hc)
    
class Generator_SC(nn.Module):
    def __init__(self, z_dim, c_dim, s_dim):
        super(Generator_SC, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE*2)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        conv_intermediate = nn.Conv2d(GEN_SIZE*4 + c_dim + GEN_SIZE*2, GEN_SIZE*4, 3, 1, padding=1)
        nn.init.xavier_uniform(conv_intermediate.weight.data, 1.)
        
        conv_s1 = nn.Conv2d(s_dim, GEN_SIZE, kernel_size=4, stride=2, padding=1)
        conv_s2 = nn.Conv2d(GEN_SIZE, GEN_SIZE*2, kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform(conv_s1.weight.data, 1.)
        nn.init.xavier_uniform(conv_s2.weight.data, 1.)
        
        self.model1 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*16, stride=2),
            ResBlockGenerator(GEN_SIZE*16, GEN_SIZE*8, stride=2),
            ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, stride=2),
            # nn.BatchNorm2d(GEN_SIZE*4),
            nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
            nn.ReLU()
            )
        self.model2 = nn.Sequential(
            conv_intermediate,
            ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, stride=2),
            ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, stride=2),
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            self.final,
            nn.Tanh())
        self.model_s = nn.Sequential(
            conv_s1,
            # nn.BatchNorm2d(GEN_SIZE),
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            conv_s2,
            # nn.BatchNorm2d(GEN_SIZE*2),
            nn.InstanceNorm2d(GEN_SIZE*2, affine=True),
            nn.ReLU()
            )

    def forward(self, z, c, s):
        h = self.model1(self.dense(z).view(-1, GEN_SIZE*16, 4, 4))
        s = self.model_s(s)
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), h.size(2), h.size(3))
        hcs = torch.cat([h, c, s], dim=1)
        return self.model2(hcs)
# class Generator_SC_2(nn.Module):
#     def __init__(self, z_dim, c_dim, s_dim):
#         super(Generator_SC_2, self).__init__()
#         self.z_dim = z_dim

#         self.dense = nn.Linear(self.z_dim, 16 * 16 * GEN_SIZE*2)
#         self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        
#         conv_s1 = nn.Conv2d(s_dim, GEN_SIZE, kernel_size=4, stride=2, padding=1)
#         conv_s2 = nn.Conv2d(GEN_SIZE, GEN_SIZE*2, kernel_size=4, stride=2, padding=1)
#         conv_s3 = nn.Conv2d(GEN_SIZE*2, GEN_SIZE*4, kernel_size=4, stride=2, padding=1)
#         conv_intermediate = nn.Conv2d(GEN_SIZE*4 + c_dim, GEN_SIZE*4, 3, 1, padding=1)
#         conv_1 = nn.Conv2d(GEN_SIZE*4+GEN_SIZE*2, GEN_SIZE*8, kernel_size=3, stride=1, padding=1)

#         nn.init.xavier_uniform(conv_intermediate.weight.data, 1.)
#         nn.init.xavier_uniform(self.dense.weight.data, 1.)
#         nn.init.xavier_uniform(self.final.weight.data, 1.)
#         nn.init.xavier_uniform(conv_s1.weight.data, 1.)
#         nn.init.xavier_uniform(conv_s2.weight.data, 1.)
#         nn.init.xavier_uniform(conv_s3.weight.data, 1.)
#         nn.init.xavier_uniform(conv_1.weight.data, 1.)

#         self.model1 = nn.Sequential(
#             conv_1,
#             ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, stride=2),
#             nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
#             nn.ReLU(),
#             )
#         self.model2 = nn.Sequential(
#             conv_intermediate,
#             ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, stride=2),
#             ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, stride=2),
#             nn.InstanceNorm2d(GEN_SIZE, affine=True),
#             nn.ReLU(),
#             self.final,
#             nn.Tanh())
#         self.model_s = nn.Sequential(
#             conv_s1,
#             nn.InstanceNorm2d(GEN_SIZE, affine=True),
#             nn.ReLU(),
#             conv_s2,
#             nn.InstanceNorm2d(GEN_SIZE*2, affine=True),
#             nn.ReLU(),
#             conv_s3,
#             nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
#             nn.ReLU()
#             )
#     def forward(self, z, c, s):
#         s = self.model_s(s)
#         z = self.dense(z).view(s.size(0), -1, s.size(2), s.size(3))
#         h = torch.cat([s, z], dim=1)
#         h = self.model1(h)
#         c = c.unsqueeze(2).unsqueeze(3)
#         c = c.expand(c.size(0), c.size(1), h.size(2), h.size(3))
#         # print(type(c))
#         # print(type(h))
#         h = torch.cat([h, c], dim=1)
#         return self.model2(h)
class Generator_SC_2(nn.Module):
    def __init__(self, z_dim, c_dim, s_dim):
        super(Generator_SC_2, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 16 * 16 * GEN_SIZE*2)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        
        conv_s1 = nn.Conv2d(s_dim, GEN_SIZE, kernel_size=4, stride=2, padding=1)
        conv_s2 = nn.Conv2d(GEN_SIZE, GEN_SIZE*2, kernel_size=4, stride=2, padding=1)
        conv_s3 = nn.Conv2d(GEN_SIZE*2, GEN_SIZE*4, kernel_size=4, stride=2, padding=1)
        conv_intermediate = nn.Conv2d(GEN_SIZE*4 + c_dim, GEN_SIZE*4, 3, 1, padding=1)
#         conv_1 = nn.Conv2d(GEN_SIZE*4+GEN_SIZE*2, GEN_SIZE*8, kernel_size=3, stride=1, padding=1)
        conv_1 = nn.Conv2d(GEN_SIZE*4+self.z_dim, GEN_SIZE*8, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_uniform(conv_intermediate.weight.data, 1.)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        nn.init.xavier_uniform(conv_s1.weight.data, 1.)
        nn.init.xavier_uniform(conv_s2.weight.data, 1.)
        nn.init.xavier_uniform(conv_s3.weight.data, 1.)
        nn.init.xavier_uniform(conv_1.weight.data, 1.)

        self.model1 = nn.Sequential(
            conv_1,
            ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, stride=2),
            nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
            nn.ReLU(),
            )
        self.model2 = nn.Sequential(
            conv_intermediate,
            ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, stride=2),
            ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, stride=2),
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            self.final,
            nn.Tanh())
        self.model_s = nn.Sequential(
            conv_s1,
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            conv_s2,
            nn.InstanceNorm2d(GEN_SIZE*2, affine=True),
            nn.ReLU(),
            conv_s3,
            nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
            nn.ReLU()
            )

    def forward(self, z, c, s):
        s = self.model_s(s)
        z = z.unsqueeze(2).unsqueeze(3)
        z = z.expand(z.size(0), z.size(1), s.size(2), s.size(3))
#         z = self.dense(z).view(s.size(0), -1, s.size(2), s.size(3))
        h = torch.cat([s, z], dim=1)
        h = self.model1(h)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), h.size(2), h.size(3))
        # print(type(c))
        # print(type(h))
        h = torch.cat([h, c], dim=1)
        return self.model2(h)

class Generator_SC_3(nn.Module):
    def __init__(self, z_dim, c_dim, s_dim):
        super(Generator_SC_3, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        
        conv_s1 = nn.Conv2d(s_dim, GEN_SIZE, kernel_size=4, stride=2, padding=1)
        conv_s2 = nn.Conv2d(GEN_SIZE, GEN_SIZE*2, kernel_size=4, stride=2, padding=1)
        conv_s3 = nn.Conv2d(GEN_SIZE*2, GEN_SIZE*4, kernel_size=4, stride=2, padding=1)
        conv_s4 = nn.Conv2d(GEN_SIZE*4, GEN_SIZE*8, kernel_size=4, stride=2, padding=1)

        nn.init.xavier_uniform(self.final.weight.data, 1.)
        nn.init.xavier_uniform(conv_s1.weight.data, 1.)
        nn.init.xavier_uniform(conv_s2.weight.data, 1.)
        nn.init.xavier_uniform(conv_s3.weight.data, 1.)
        nn.init.xavier_uniform(conv_s4.weight.data, 1.)

        self.model1 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*8+self.z_dim+self.c_dim, GEN_SIZE*8, stride=2),
            ResBlockGenerator(GEN_SIZE*8, GEN_SIZE*4, stride=2),
            ResBlockGenerator(GEN_SIZE*4, GEN_SIZE*2, stride=2),
            ResBlockGenerator(GEN_SIZE*2, GEN_SIZE, stride=2),
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            self.final,
            nn.Tanh())
        self.model_s = nn.Sequential(
            conv_s1,
            nn.InstanceNorm2d(GEN_SIZE, affine=True),
            nn.ReLU(),
            conv_s2,
            nn.InstanceNorm2d(GEN_SIZE*2, affine=True),
            nn.ReLU(),
            conv_s3,
            nn.InstanceNorm2d(GEN_SIZE*4, affine=True),
            nn.ReLU(),
            conv_s4,
            nn.InstanceNorm2d(GEN_SIZE*8, affine=True),
            nn.ReLU()
            )

    def forward(self, z, c, s):
        s = self.model_s(s)
        z = z.unsqueeze(2).unsqueeze(3)
        z = z.expand(z.size(0), z.size(1), s.size(2), s.size(3))
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), s.size(2), s.size(3))
        h = torch.cat([s, z, c], dim=1)
        h = self.model1(h)
        return h

class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE*2, stride=2),
                ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride=2),
                ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride=2),
                ResBlockDiscriminator(DISC_SIZE*8, DISC_SIZE*16, stride=2),
                ResBlockDiscriminator(DISC_SIZE*16, DISC_SIZE*16),
                nn.ReLU(),
                nn.AvgPool2d(4),
            )
        self.fc = nn.Linear(DISC_SIZE*16, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)
        self.classify = nn.Linear(DISC_SIZE*16, c_dim)
        nn.init.xavier_uniform(self.classify.weight.data, 1.)
        self.m = nn.Tanh()
    def forward(self, x):
        h = self.model(x).view(-1,DISC_SIZE*16)
        # return self.fc(h), self.classify(h)
        return self.fc(h), self.classify(h)

class Generator_CNN(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim, c_dim):
        super(Generator_CNN, self).__init__()

        self.input_height = 128
        self.input_width = 128
        self.input_dim = z_dim
        self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024*4*4),
            nn.BatchNorm1d(1024*4*4),
            nn.ReLU(),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256+c_dim, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input, c):
        x = self.fc(input)
        x = x.view(-1, 1024, 4, 4)
        x = self.deconv1(x)
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.deconv2(x)

        return x

class Discriminator_CNN(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, c_dim=5):
        super(Discriminator_CNN, self).__init__()
        image_size=128
        conv_dim=64
        repeat_num=6
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
        self.m = nn.Tanh()
    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        # return out_real.squeeze(), out_aux.squeeze()
        return out_real.squeeze(), out_aux.squeeze()
    # def forward(self, x):
    #     self.gpu_ids=[0,1]
    #     h = nn.parallel.data_parallel(self.main, x, self.gpu_ids)
    #     out_real = nn.parallel.data_parallel(self.conv1, h, self.gpu_ids)
    #     out_aux = nn.parallel.data_parallel(self.conv2, h, self.gpu_ids)
    #     out_aux = nn.parallel.data_parallel(self.m, out_aux, self.gpu_ids)
    #     # h = self.main(x)
    #     # out_real = self.conv1(h)
    #     # out_aux = self.conv2(h)
    #     # return out_real.squeeze(), out_aux.squeeze()
    #     return out_real.squeeze(), self.m(out_aux).squeeze()

class Segmentor(nn.Module):
    """Segmentor."""
    def __init__(self, conv_dim=64, repeat_num=4):
        super(Segmentor, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 7, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.LogSoftmax())
        # layers.append(nn.Softmax2d())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)