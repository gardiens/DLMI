import torch.nn as nn
class mainmodel(nn.Module):
    def __init__(self,model,linear_probing):
        super(mainmodel, self).__init__()
        self.feature_extractor = model
        self.linear_probing=linear_probing
    def forward(self,x):
        x=self.feature_extractor(x)
        x=self.linear_probing(x)
        return x
    
import torch 
class baseLine(nn.Module):
    def __init__(self,device):
        super(baseLine, self).__init__()
        self.feature_extractor= torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.feature_extractor.eval()
        self.linear_probing=torch.nn.Sequential(torch.nn.Linear(self.feature_extractor.num_features, 1),
                                     torch.nn.Sigmoid()).to(device)
    def forward(self,x):
        x=self.feature_extractor(x)
        x=self.linear_probing(x)
        return x