import torch.nn as nn
from timm import create_model, list_models

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.num_classes = args.num_classes
        self.load_model = args.load_model
        if self.load_model:
            # list_models('resnet*', pretrained=True)
            self.backbone = create_model('resnet18', pretrained=True, num_classes=args.num_classes)

    def forward(self, x):
        if self.load_model:
            x = self.backbone(x)
        return x
    

class Classifier2(nn.Module):
    def __init__(self, load_model, num_classes):
        super(Classifier2, self).__init__()

        self.num_classes = num_classes
        self.load_model = load_model
        if self.load_model:
            self.model = create_model(self.load_model, pretrained=True, num_classes=self.num_classes)

    def forward(self, x):
        if self.load_model:
            # x = self.backbone(x)
            # x = self.head(x)
            x = self.model(x)
        return x


class AgeClassifier(nn.Module):
    def __init__(self, load_model, num_classes):
        super(AgeClassifier, self).__init__()

        self.num_classes = num_classes
        self.load_model = load_model
        if self.load_model:            
            self.backbone = create_model(self.load_model, pretrained=True)
            self.head = nn.Sequential(nn.Linear(1000, 500),
                                       nn.ReLU(),
                                       nn.Linear(500, 250),
                                       nn.ReLU(),
                                       nn.Linear(250, 125),
                                       nn.ReLU(),
                                       nn.Linear(125, 50),
                                       nn.ReLU(),
                                       nn.Linear(50, self.num_classes)
                                       )            

    def forward(self, x):
        if self.load_model:
            x = self.backbone(x)
            x = self.head(x)
        return x