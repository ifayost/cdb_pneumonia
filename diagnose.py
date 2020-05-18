import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from tqdm import tqdm
from transform_images import crop_images
from captum.attr import Occlusion
import copy

plt.style.use('seaborn')
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 12

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = "./test"

if "diagnoses" not in os.listdir():
    os.mkdir("diagnoses")

if len(sys.argv) == 2:
    if sys.argv[1] == 'crop':

        shapes = []
        list_path = []
        for j in os.listdir(PATH):
            if j != '.DS_Store':
                try:
                    im_path = os.path.join(PATH, j)
                    list_path.append(im_path)
                    im = Image.open(im_path)
                    size = im.size
                    shapes.append(size)
                except:
                    print("Error with croping:", im_path)
        shapes = np.array(shapes)

        print("Crop images")
        for i in tqdm(list_path):
            im = Image.open(i)
            im = crop_images(im, shapes)
            im.save(i)
    else:
        print("Invalid argument")

class_names_PN = ["Sano", "Neumonia"]
class_names_BV = ["Bacteriana", "Vírica"]

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

image_datasets = CustomDataSet(PATH, data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1)

vgg16PN = torchvision.models.vgg16(pretrained=False)
vgg16PN.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
vgg16PN = nn.Sequential(nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), vgg16PN)
vgg16VB = copy.deepcopy(vgg16PN)

if device.type == "cpu":
    vgg16PN.load_state_dict(torch.load('./weights/vgg16+1conv_224x224_e5.pt', map_location=torch.device('cpu')))
else:
    vgg16PN.load_state_dict(torch.load('./weights/vgg16+1conv_224x224_e5.pt'))
vgg16PN = vgg16PN.to(device)
vgg16PN = vgg16PN.eval()

if device.type == "cpu":
    vgg16VB.load_state_dict(torch.load('./weights/vgg16+VB_224x224.pt', map_location=torch.device('cpu')))
else:
    vgg16VB.load_state_dict(torch.load('./weights/vgg16+VB_224x224.pt'))
vgg16VB = vgg16VB.to(device)
vgg16VB = vgg16VB.eval()

def prediction(inp):
    outputs = vgg16PN(inp)
    probN = F.softmax(outputs, 1)[0][0].detach().cpu().numpy()
    if int(np.round(probN)) == 0:
        outputs = vgg16VB(inp)
        probsVB = F.softmax(outputs, 1)[0].detach().cpu().numpy()
    else:
      probsVB = (1-probN)/2 * np.ones(2)
    probs = np.append(probN, probsVB)
    return probs/np.sum(probs)

occlusion = Occlusion(vgg16PN)
for inp, filename in tqdm(zip(dataloaders, image_datasets.total_imgs)):
    inp = inp.to(device)
    probs = prediction(inp)
    attributions_ig = occlusion.attribute(inp,
                                          strides=(1, 10, 10),
                                          target=0,
                                          sliding_window_shapes=(1, 15, 15),
                                          baselines=0)

    fig = plt.figure(figsize=(18, 6), dpi=200)
    ax1 = fig.add_subplot(131)
    ax1.imshow((inp.squeeze().cpu().detach().numpy()), cmap='viridis')
    ax1.axis('off');

    ax2 = fig.add_subplot(132)
    ax2.imshow((inp.squeeze().cpu().detach().numpy()), cmap='Greys')
    im = ax2.imshow(attributions_ig.squeeze().cpu().detach().numpy(), alpha=0.6, cmap='YlGnBu', interpolation='sinc')
    ax2.axis('off');

    ax3 = fig.add_subplot(133)
    ax3.bar(range(3), probs)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(["Normal", "Bacteriana", "Virica"])

    box2 = np.array(ax2.get_position())
    cbar_ax = fig.add_axes([box2[0, 0] - 0.03, box2[0, 1], 0.01, box2[1, 1] - box2[0, 1]])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(os.path.join("diagnoses", filename))