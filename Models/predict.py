from utils import test_dataloader
from models import NeuralNetworkClassifier
import numpy as np
import torch
import os

class Predict(NeuralNetworkClassifier):
    def __init__(self, model_name: str= ""):
        if model_name.upper() in ["ALEXNET", "SQUEEZENET", "GOOGLENET", "RESNET"]:
            self.classifier = NeuralNetworkClassifier(model_name)
            self.classifier.initialize_model(num_classes=3)
        self.name = self.classifier.model._get_name()
        self.device = self.classifier.device

    def load_weights(self, root_path: str) -> None:
        weights_path = os.path.join(root_path, self.name )
        if not os.path.exists(weights_path):
            raise FileNotFoundError("The path {} contains no weights file".format(weights_path))
        weights = torch.load(weights_path+"/weights.pth")
        self.classifier.model.load_state_dict(weights)
        print("\n***Weights successfully loaded***\n")

    def test_set_eval(self, dataloader):

        with torch.no_grad():
            cnt = 0; corr = 0
            for data in dataloader:
                imgs = [i.to(self.device) for i in data[:-1]]
                
                label = data[-1].to(self.device)
                prediction = self.classifier.model(*imgs)
                pred = torch.argmax(prediction)
                print(prediction, pred, "\t", label)
                if(pred == label):
                    print(pred, label)
                    corr += 1
                cnt += 1
        print(corr/ cnt)


if __name__ == "__main__":
    predictor = Predict(model_name="resnet")
    predictor.load_weights("Weights")
    test_dataloader = test_dataloader("images/test", batch_size=1)
    predictor.test_set_eval(test_dataloader)


