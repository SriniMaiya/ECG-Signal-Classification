import os
from PIL import Image
from cv2 import transform
from models import NeuralNetworkClassifier
import torch
from torchvision import transforms

classifier = NeuralNetworkClassifier("resnet")
classifier.initialize_model(num_classes=3)
weights = torch.load("Weights/ResNet/best_weights.pth")
classifier.model.load_state_dict(weights)

signals = ["ARR", "CHF", "NSR"]

for i, sig in enumerate(signals):
    img_loc = os.listdir(os.path.join("images/test", sig))
    print(sig)
    for img in img_loc:
        img = os.path.join("images/test",sig, img)
        img = Image.open(img)
        totensor = transforms.Compose([transforms.ToTensor()])
        img = totensor(img)
        img = torch.unsqueeze(img, 0).cuda()
        print(torch.argmax(classifier.model(img)))


        
