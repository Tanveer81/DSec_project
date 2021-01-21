from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
from efficientnet_pytorch import EfficientNet
import torchattacks


SEED = 512
DATA_ROOT_PATH = '../alaska2-image-steganalysis'

# For reproducibility all seeds are set.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# Creating Dateset
print('creating dataset')
dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD', 'SteganoGAN']):
    print(label, kind)
#     for path in glob(f'../alaska2-image-steganalysis/{kind}/*.jpg'):
    for path in sorted(os.listdir(f'../alaska2-image-steganalysis/{kind}/')):
        path = '../alaska2-image-steganalysis/' + kind + '/' + path
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })
    print(path)
random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

gkf = GroupKFold(n_splits=32)

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
    
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

# Loading model
print('Load model')
def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net

net = get_net().cuda()

checkpoint = torch.load('../best-checkpoint-143epoch.bin')
net.load_state_dict(checkpoint['model_state_dict']);
net.eval();

# Augmentation
def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    

class CustomDatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(5, label) #@ Tanveer
        return image_name, image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
    

    
results = []
atks = [torchattacks.FGSM(net, eps=8/255),
        torchattacks.BIM(net, eps=8/255, alpha=2/255, steps=7),
        torchattacks.CW(net, c=1, kappa=0, steps=1000, lr=0.01),
        torchattacks.RFGSM(net, eps=8/255, alpha=4/255, steps=1),
        torchattacks.PGD(net, eps=8/255, alpha=2/255, steps=7),
        torchattacks.FFGSM(net, eps=8/255, alpha=12/255),
        torchattacks.MIFGSM(net, eps=8/255, decay=1.0, steps=5),
        torchattacks.TPGD(net, eps=8/255, alpha=2/255, steps=7),
       ]

val_dataset = CustomDatasetRetriever(
    kinds=dataset[dataset['fold'] == fold_number].kind.values,
    image_names=dataset[dataset['fold'] == fold_number].image_name.values,
    labels=dataset[dataset['fold'] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

data_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    drop_last=False,
)

target_map_function = lambda images, labels: (labels==0).long()*np.random.randint(1,4)

for atk in atks:
    if atk.attack != 'TPGD':
        atk.set_targeted_mode(target_map_function=target_map_function)
    
for step, (image_names, images, targets) in enumerate(data_loader):
    result = {'Id': [], 'Truth': [], 'Prediction': [],}
    start = time.clock() 
    if int(step) % 10 == 0:
        print(f"step: {step}, start_time: {start}")

    targets = torch.argmax(targets, dim=1)
    y_pred = net(images.cuda())
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]

    for atk in atks :
        if atk.attack != 'TPGD':
             adv_images = atk(images, targets)
        else:
             adv_images = atk(images, (targets==0).long()*np.random.randint(1,4))
        y_pred_attk = net(adv_images)
        y_pred_attk = 1 - nn.functional.softmax(y_pred_attk, dim=1).data.cpu().numpy()[:,0]
        try:
            result[f'{atk.attack}_prediction'].extend(y_pred_attk)
        except KeyError:
            result[f'{atk.attack}_prediction'] = y_pred_attk
        

    result['Id'].extend(image_names)
    result['Prediction'].extend(y_pred)
    result['Truth'].extend(targets.tolist())
    
    results.append(result)

    
submissions = []
for i in range(len(results)):
    submission = pd.DataFrame(results[i])
    submissions.append(submission)
submissions = pd.concat(submissions)
submissions.to_csv('../Evaluation/adv_attack/submission.csv', index=False)  
