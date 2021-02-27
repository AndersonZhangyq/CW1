"""

Template code for COMP5623M CW1 Question 2

python explore.py --image_path XX --use_pre_trained True


"""

import argparse
import torch



# Set up training arguments and parse
parser = argparse.ArgumentParser(description='Explore pre-trained AlexNet')


parser.add_argument(
    '--image_path', type=str, default="imagenet10/train_set/cat/n02123159_109.JPEG",
    help='Full path to the input image to load.')
parser.add_argument(
    '--use_pre_trained', type=bool, default=True,
    help='Load pre-trained weights?')


args = parser.parse_args()

# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("=======================================")
print("                PARAMS               ")
print("=======================================")
for arg in vars(args):
    print(F"{arg:>20} {getattr(args, arg)}")


#########################################################################
#
#        QUESTION 2.1.2 code here
#
#########################################################################


# Read in image located at args.image_path
from PIL import Image
img = Image.open(args.image_path)



# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]



# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# To see the AlexNet architecture
print(model)

model.eval()


# Pass image through a single forward pass of the network
import torchvision.transforms as transforms
data_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
with torch.no_grad():
    data = data_transform(img)
    data = data.view(1, *data.shape)
    output = model(data)


# layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]

#########################################################################
#
#        QUESTION 2.1.3
#
#########################################################################

def extract_filter(conv_layer_idx, model):
    """ Extracts a single filter from the specified convolutional layer,
        zero-indexed where 0 indicates the first conv layer.

        Args:
            conv_layer_idx (int): index of convolutional layer
            model (nn.Module): PyTorch model to extract from

    """

    # Extract filter
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    return model.features[conv_layer_idx].weight.data

#########################################################################
#
#        QUESTION 2.1.4
#
#########################################################################


def extract_feature_maps(input, model):
    """ Extracts the all feature maps for all convolutional layers.

        Args:
            input (Tensor): input to model
            model (nn.Module): PyTorch model to extract from

    """

    # Extract all feature maps
    # Hint: use conv_layer_indices to access
    with torch.no_grad():
        feature_maps = []
        x = input
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                feature_maps.append(x)
    return feature_maps

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
output_path = writer.get_logdir()
import os, math, random
from torchvision import utils

with torch.no_grad():
    # visualize weight
    selected_channel = [-1, -1, -1, -1, -1]
    for layer_idx, channel in zip(conv_layer_indices, selected_channel):
        weight = extract_filter(layer_idx, model)
        n, c, h, w = weight.shape
        if channel == -1:
            channel = random.randint(0, c - 1)
        weight = weight[:, channel, :, :].unsqueeze(1)
        nrow = int(math.sqrt(n))
        print(h, w)
        grid = utils.make_grid(weight, nrow=nrow, normalize=True, scale_each=True)
        utils.save_image(weight, os.path.join(output_path, f"conv_{layer_idx}.png"), nrow=nrow, normalize=True, scale_each=True)
        writer.add_image("conv filter", grid, layer_idx)


    # visualzie feature map
    feature_map = extract_feature_maps(data, model)
    for idx, fm in enumerate(feature_map):
        n, c, h, w = fm.shape
        fm = fm.view(n * c, -1, h, w)
        nrow = int(math.sqrt(n * c))
        print(h, w)
        grid = utils.make_grid(fm, nrow=nrow, normalize=True, scale_each=True)
        utils.save_image(fm, os.path.join(output_path, f"feature_map_{idx}.png"), nrow=nrow, normalize=True, scale_each=True)
        writer.add_image("feature map", grid, idx)

writer.close()
