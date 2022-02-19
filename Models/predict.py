from cgi import test
from utils import test_dataloader
from models import NeuralNetworkClassifier
import numpy as np
import torch
import os

class Predict(NeuralNetworkClassifier):
    def __init__(self, model_name: str= ""):
        super(NeuralNetworkClassifier, self).__init__()
        if model_name.upper() in ["ALEXNET", "SQUEEZENET", "GOOGLENET"]:
            self.classifier = NeuralNetworkClassifier(model_name)
            self.classifier.initialize_model(num_classes=3)
        self.name = self.classifier.model._get_name()
        self.device = self.classifier.device
    def load_weights(self, root_path: str) -> None:
        weights_path = os.path.join(root_path, self.name )
        if not os.path.exists(weights_path):
            raise FileNotFoundError("The path {} contains no weights file".format(weights_path))
        weights = torch.load(weights_path+"/best_weights.pth")
        self.classifier.model.load_state_dict(weights)
        print("\n***Weights successfully loaded***")

    def predict_from_scalogram(self, dataloader):
        pred_ARR, pred_NSR, pred_CHF = 0, 0, 0

        for img in next(iter(dataloader)):
            img, label= img
            img = img.to(self.device); label = label.to(self.device)
            prediction = self.classifier.model(img).cuda()    
            print(prediction)
            print(img)

if __name__ == "__main__":
    predictor = Predict(model_name="alexnet")
    predictor.load_weights("Weights")
    dataloader = test_dataloader("images/test", batch_size=1)
    predictor.predict_from_scalogram(dataloader)
        
