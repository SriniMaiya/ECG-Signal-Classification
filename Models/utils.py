from torchvision import datasets, transforms
import numpy as np
import torch
import os
'''
Utility File for loading the training and test dataset with other preprocessing
'''

def train_dataloader(input_size, dataset_path:str, batch_size:int = 32 ):
    '''Loads the train and validation images. The images are in tensor format. All the tensors are packed as a dictionary.
        The images are resized, converted to tensor and normalized.'''
    data_transforms = {
                        'train':transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                        ]),

                        'val': transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])    
                        ]),
                        }
    

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), data_transforms[x]) for x in ["train", "val"] }
    dataloader_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers = 1) for x in ["train", "val"]}
    return dataloader_dict

def test_dataloader(testset_path:str, batch_size:int = 32):
    '''Dataloader for testing of the model'''
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor()
                                    ])

    dataset = datasets.ImageFolder(testset_path, transform)
    dataloader_dict = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_workers = 2)
    
    return dataloader_dict


