from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import time
import os
import copy
from utils import train_dataloader
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


class NeuralNetworkClassifier:
    def __init__(self, model_name:str = "AlexNet"):
        if model_name.upper() == "alexnet".upper():
            self.model = models.alexnet(pretrained=True, progress=True)
            self.name = self.model._get_name()
        elif model_name.upper() == "squeezenet".upper():
            self.model = models.squeezenet1_1(pretrained=True, progress=True)
            self.name = self.model._get_name()
        elif model_name.upper() == "googlenet".upper():
            self.model = models.googlenet(pretrained=True, progress=True)
            self.name = self.model._get_name()

        self.input_size = 224
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialize_model(self,num_classes:int = 3):



        if self.name.upper() == "alexnet".upper():
            n_feat = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(n_feat, num_classes)

        elif self.name.upper() == "squeezenet".upper():
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = num_classes

        elif self.name.upper() == "googlenet".upper():
            n_feat = self.model.fc.in_features
            self.model.fc = nn.Linear(n_feat, num_classes)
        
        self.model = self.model.to(self.device)


    def start_training(self, dataloaders, num_epochs, save_weights:bool = True, weights_path:str="", feature_extract:bool = True, learning_rate:float=0.001 ):
        params_to_update = []
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = True
                params_to_update.append(param)

        self.optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        since = time.time()
        history = {"train":{"loss":[], "acc":[]}, "val": {"loss":[], "acc":[]}}

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print("--"*10)

            for phase in ["train", "val"]:
                if phase =="train":
                    self.model.train()
                elif phase == "val":
                    self.model.eval()
                
                running_loss = 0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase== "train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = np.array([(running_corrects.double() / len(dataloaders[phase].dataset)).cpu().detach()])[0]
                print("{} loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase == "val":
                    history["val"]["loss"].append(epoch_loss)
                    history["val"]["acc"].append(epoch_acc)
                else:
                    history["train"]["loss"].append(epoch_loss)
                    history["train"]["acc"].append(epoch_acc)

        print()

        time_elapsed = time.time()- since

        print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed%60))
        print("Best Val Accuracy: {:.4f}".format(best_acc))

        if save_weights:
            weights_path = os.path.join(weights_path, self.name)
            os.makedirs(weights_path, exist_ok=True)
            torch.save(self.model.state_dict(), weights_path+"/weights.pth")
            torch.save(best_model_wts, weights_path+"/best_weights.pth")

        return self.model, history
if __name__ == "__main__":

    classifier = NeuralNetworkClassifier(model_name="alexnet")
    dataloader_dict = train_dataloader(input_size=224, dataset_path="images", batch_size=32)
    classifier.initialize_model(num_classes=3)
    model, history = classifier.start_training(dataloaders=dataloader_dict, num_epochs=30, save_weights= True, 
                                weights_path="Weights",  feature_extract=True, learning_rate=0.001)
    torch.cuda.empty_cache()
    
    # print(history)




