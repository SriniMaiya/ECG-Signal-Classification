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

''' NeuralNetworkClassifier: A Class which defines 4 models, AlexNet, GoogLeNet, SqueezeNet and ResNet which
                            can be initialized and trained.
    Input:  model_name : "AlexNet", "SqueezeNet", "GoogLeNet", "ResNet" [string]

'''


class NeuralNetworkClassifier:
    def __init__(self, model_name:str = "AlexNet"):
        '''Creates a model from 'torchvision' according to user input.
        '''
        #Create model and store the model name for future references
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
        #Input size of the images
        self.input_size = 224
        self.scheduler = None                                                               #Scheduler (initialized later)
        self.class_weights = None                                                           #Weights for correcting the imbalance in the dataset
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)                     #Type of criterion to calculate loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        # Available device
        self.params = None

    def initialize_model(self,num_classes:int = 3):
        '''Model is initialized with 3 classes(ARR; CHF; NSR)
           Additional fine-tuning of the model to prevent overfitting'''
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

        # Deploy the model to the available device (GPU or CPU training)
        self.model = self.model.to(self.device)


    def start_training(self, dataloaders, num_epochs, learning_rate:float):
        '''Train the model
            Input  : dataloaders    : pytorch dataloader(defined in utils.py
                     num_epochs     : number of epochs to the network on
                     learning rate  : Learning rate for the chosen model

            Output : [trained model, model history(loss and accuracy curves), weights, best weights] )  '''

        for param in self.model.parameters():
            param.requires_grad = True                                                                  #Set gradients for training
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)                  #Create the optimizer
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.3, verbose=True)      #Scheduler creation. Decreases learning rate by 33.33%
                                                                                                        #in every 8 epochs
        since = time.time()
        history = {"train":{"loss":[], "acc":[]}, "val": {"loss":[], "acc":[]}}                         #Initialize history to null

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print("--"*10)

            for phase in ["train", "val"]:
                if phase =="train":                                                                     #train the model
                    self.model.train()
                elif phase == "val":                                                                    #validate every step
                    self.model.eval()

                running_loss = 0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:                                               #Dataloader(for both training and validation)
                    # If the data for the epoch is skewed to a class, divide by 1 to decrease it's
                    # relevance. Prevents overfitting
                    weights = np.array([sum((labels.numpy() == t)) for t in [0,1,2] ])                  #Store number of images of each type of signal
                    weights = weights + 0.00000001                                                      #Addition to avoid 1/0 Error
                    weights = 1./ np.array(weights)                                                     # 1 / each element of array

                    #Convert the above weights to tensors according type of training(CPU/GPU)
                    if self.device.type.startswith("cuda"):
                        self.class_weights = torch.FloatTensor(weights).cuda()
                    else:
                        self.class_weights = torch.FloatTensor(weights)
                    #Pass the weights to the Loss Criterion
                    self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

                    inputs = inputs.to(self.device)                                             #Deploy the image to device
                    labels = labels.to(self.device)                                             #Deploy the labels to device

                    optimizer.zero_grad()                                                       #Set the gradients to zero once optimized

                    with torch.set_grad_enabled(phase== "train"):
                        outputs = self.model(inputs)                                            #Predict
                        #Modification for googlenet
                        if self.name == "GoogLeNet" and phase == "train":
                            outputs = outputs.logits

                        loss = self.criterion(outputs, labels)                                  #Get the predicted class
                        _, preds = torch.max(outputs, 1)

                        if phase == "train":                                                    #If train, backpropagation
                            loss.backward()
                            optimizer.step()

                    # print("Prediction:", preds,"\n")
                    running_loss += loss.item() * inputs.size(0)                                #Append loss for each substep
                    running_corrects += torch.sum(preds == labels.data)                         #Correct prediction for each substep

                # print(labels.cpu().numpy())
                if phase == "train":                                                            #If training, register the epoch with scheduler 
                    scheduler.step()
                #Loss and accuracy for the whole epoch    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = np.array([(running_corrects.double() / len(dataloaders[phase].dataset)).cpu().detach()])[0]
                print("{} loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                #Save model history for each epoch
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

            # Main is not used. Function call happens in GuiControlCommand.py
# if __name__ == "__main__":
#     torch.cuda.empty_cache()



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
