import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

class ImageDataset(Dataset):
    def __init__(self, image, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.imageData = image # Image is a Numpy Array/CV2 Image

    def __getitem__(self, index):
        item_A = self.transform(self.imageData)
        return {"A": item_A}
  
    def __len__(self):
        return 1

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 6),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7, padding=2), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class DepthEstimator:
    def __init__(self, model_dir=None, height=200, width=200, n_residual_blocks=9):
        self.img_height = height
        self.img_width = width

        self.cuda = torch.cuda.is_available()
        input_shape = (1, self.img_height, self.img_width)
        self.G_AB = GeneratorResNet(input_shape, n_residual_blocks)

        if self.cuda:
            self.G_AB = self.G_AB.cuda()

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models")

        weights_path = os.path.join(model_dir, "6Level-ex-vivo-G_AB.pth")
        self.G_AB.load_state_dict(torch.load(weights_path, map_location=None if self.cuda else torch.device('cpu')))
        self.G_AB.eval()

        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.transforms_ = [transforms.ToTensor()]

    def generateDepth(self, image):
        """Generate a depth map from a PIL Image.

        Args:
            image: PIL Image (grayscale).

        Returns:
            PIL Image containing the depth map.
        """
        image_array = np.array(image)

        dataloader = DataLoader(
            ImageDataset(image_array, transforms_=self.transforms_),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        for batch in dataloader:
            real_A = Variable(batch["A"].type(self.Tensor))
            fake_B = self.G_AB(real_A)

        img_tensor = (fake_B.squeeze(0).cpu().detach() * 255).byte()
        depth_image = transforms.ToPILImage()(img_tensor)
        return depth_image
