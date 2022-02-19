from torchvision import datasets, transforms
import numpy as np
import torch
import os


def train_dataloader(input_size, dataset_path:str, batch_size:int = 64 ):

    data_transforms = {
                        'train':transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                        'val': transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                        }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x]) for x in ["train", "val"] }
    dataloader_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size, shuffle = True, num_workers = 4) for x in ["train", "val"]}
    return dataloader_dict

def test_dataloader(testset_path:str, batch_size:int = 32):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()
                                    ])

    dataset = datasets.ImageFolder(testset_path, transform)
    dataloader_dict = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers = 4, pin_memory=True)
    return dataloader_dict


        