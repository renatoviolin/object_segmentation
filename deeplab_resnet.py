# %%
import os
import cv2
import numpy as np
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import imutils
import uuid
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dlab.to(device)

# %%
LABEL_NAMES = np.asarray([
    'all masks', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


def decode_segmap(image, source, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r_all = np.zeros_like(image).astype(np.uint8)
    g_all = np.zeros_like(image).astype(np.uint8)
    b_all = np.zeros_like(image).astype(np.uint8)
    maps = []

    for l in range(nc):
        idx = image == l
        r_all[idx] = label_colors[l, 0]
        g_all[idx] = label_colors[l, 1]
        b_all[idx] = label_colors[l, 2]

        r_unique = np.zeros_like(image).astype(np.uint8)
        g_unique = np.zeros_like(image).astype(np.uint8)
        b_unique = np.zeros_like(image).astype(np.uint8)
        r_unique[idx] = label_colors[l, 0]
        g_unique[idx] = label_colors[l, 1]
        b_unique[idx] = label_colors[l, 2]

        maps.append(np.stack([r_unique, g_unique, b_unique], axis=2))

    rgb = np.stack([r_all, g_all, b_all], axis=2)
    return rgb, maps


def segment(path, dev=device):
    img = Image.open(path)
    trf = T.Compose([
        T.Resize(480),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    inp = trf(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = dlab(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    mask, all_maps = decode_segmap(om, path)
    w = inp.shape[3]
    h = inp.shape[2]
    return mask, img.resize((w, h)), all_maps, np.unique(om)


def generate_segments(input_path, output_folder):
    mask, img, all_maps, class_idx = segment(input_path)
    out_files = []
    for i in class_idx:
        file_id = str(uuid.uuid1()).split('-')[0]
        out = os.path.join(output_folder, f'{i}_{file_id}.png')
        out_files.append(out)
        if i == 0:
            added_image = cv2.addWeighted(np.array(img), 0.5, mask, 1.3, 0)
            cv2.imwrite(out, cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB))
        else:
            added_image = cv2.addWeighted(np.array(img), 0.5, all_maps[i], 1.3, 0)
            cv2.imwrite(out, cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB))
    return out_files, list(LABEL_NAMES[class_idx])
