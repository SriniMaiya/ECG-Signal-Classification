from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import time
import copy
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)


class NeuralNetworkClassifier:
    def __init__(self, model_name:str = "AlexNet"):
        if model_name.upper() == "alexnet".upper():
            self.model = models.alexnet(pretrained=True, progress=True)
            self.name = self.model._get_name() 
        elif model_name.upper() == "squeezenet".upper():
            self.model = models.squeezenet1_1(pretrained=True, progress=True)
            self.name = self.model._get_name()
        elif model_name.upper() == "googlenet".upper():
            self.model = models.googlenet(pretrained=False, progress=False, init_weights=True)
            self.name = self.model._get_name()
        elif model_name.upper() == "resnet".upper():
            self.model = models.resnet18(pretrained=True, progress=False)
            self.name = self.model._get_name()

        self.input_size = 224
        # self.optimizer = None
        self.scheduler = None
        self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.params = None
    
    def initialize_model(self,num_classes:int = 3):
        if self.name.upper() == "alexnet".upper():
            self.model.classifier[0] = nn.Dropout(p=0.65, inplace=True)
            self.model.classifier[3] = nn.Dropout(p=0.6, inplace=False)
            n_feat = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(n_feat, num_classes)



        elif self.name.upper() == "squeezenet".upper():
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = num_classes


        elif self.name.upper() == "googlenet".upper():
            self.model.dropout = nn.Dropout(p=0.6, inplace=False)
            n_feat = self.model.fc.in_features
            self.model.fc = nn.Linear(n_feat, num_classes)


        elif self.name.upper() == "resnet".upper():
            self.model.fc = nn.Sequential(
                        nn.Dropout(p=0.65),
                        nn.Linear(in_features=512, out_features=num_classes, bias=True))





                                     
        # print(self.model.eval())
        self.model = self.model.to(self.device)


    def start_training(self, dataloaders, num_epochs, learning_rate:float):
        for param in self.model.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.3, verbose=True)

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
                    weights = np.array([sum((labels.numpy() == t)) for t in [0,1,2] ])
                    weights = weights + 0.00000001
                    weights = 1./ np.array(weights) 
                    self.class_weights = torch.FloatTensor(weights).cuda()
                    self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                        
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # print("Labels:", labels)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase== "train"):
                        outputs = self.model(inputs)
                        if self.name == "GoogLeNet" and phase == "train":
                            # outputs = outputs[0]
                            outputs = outputs.logits
                        loss = self.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # print("Prediction:", preds,"\n")
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                print(labels.cpu().numpy())  
                if phase == "train":
                    scheduler.step()
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


        torch.cuda.empty_cache()
        return self.model, history, self.model.state_dict(), best_model_wts
if __name__ == "__main__":
    torch.cuda.empty_cache()



    # # --> GOOGLENET: TO BE IMPROVED
    # model = NeuralNetworkClassifier("googlenet")
    # dataloader = train_dataloader(input_size = 224, dataset_path="images", batch_size=32) # add Qcombox for batch size
    # model.initialize_model(num_classes=3)
    # model, history = model.start_training(dataloaders=dataloader, num_epochs=20,
    #                                 learning_rate=0.003, save_weights=False, weights_path="Weights")

    # #--> ALEXNET : Getting good res
    # model = NeuralNetworkClassifier("alexnet")
    # dataloader = train_dataloader(input_size = 224, dataset_path="images", batch_size=16) # add Qcombox for batch size
    # model.initialize_model(num_classes=3)
    # model, history = model.start_training(dataloaders=dataloader, num_epochs=30,
    #                                 learning_rate=0.003, save_weights=True, weights_path="Weights")


    # #-->  SQUEEZENET: Getting good res
    # model = NeuralNetworkClassifier("squeezenet")
    # dataloader = train_dataloader(input_size = 224, dataset_path="images", batch_size=16) # add Qcombox for batch size
    # model.initialize_model(num_classes=3)
    # model, history = model.start_training(dataloaders=dataloader, num_epochs=25,
    #                                 learning_rate=0.003, save_weights=True, weights_path="Weights")



    # #--> RESNET: TO BE IMPROVED
    # model = NeuralNetworkClassifier("resnet")
    # dataloader = train_dataloader(input_size = 224, dataset_path="images", batch_size=32) # add Qcombox for batch size
    # model.initialize_model(num_classes=3)
    # model, history = model.start_training(dataloaders=dataloader, num_epochs=20,
    #                                 learning_rate=0.003, save_weights=True, weights_path="Weights")


    print( models.resnet18().eval())
