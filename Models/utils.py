from torchvision import datasets, transforms
import numpy as np
import torch
import os


def train_dataloader(input_size, dataset_path:str, batch_size:int = 32 ):

    data_transforms = {
                        'train':transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                        ]),
                        'val': transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                        ]),
                        }
    

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x]) for x in ["train", "val"] }
    dataloader_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers = 1) for x in ["train", "val"]}
    return dataloader_dict

def test_dataloader(testset_path:str, batch_size:int = 32):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()
                                    ])

    dataset = datasets.ImageFolder(testset_path, transform)
    dataloader_dict = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers = 2)
    
    return dataloader_dict


