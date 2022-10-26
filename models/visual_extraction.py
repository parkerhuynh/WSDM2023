import torch.nn as nn
import torchvision.models as models
import torch

class VisualEncoder(nn.Module):

    def __init__(self, image_embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(VisualEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])

        self.model = model
        self.fc = nn.Linear(in_features, image_embed_size)

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            x = self.model(image)
        x = self.fc(x)

        l2_norm = x.norm(p=2, dim=1, keepdim=True).detach()
        x = x.div(l2_norm)

        return x