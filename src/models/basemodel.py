import torch.nn as nn
# class mainmodel(nn.Module):
#     def __init__(self,model,linear_probing):
#         super(mainmodel, self).__init__()
#         self.feature_extractor = model
#         self.linear_probing=linear_probing
#     def forward(self,x):
#         x=self.feature_extractor(x)
#         x=self.linear_probing(x)
#         return x
    
class Linear_probingStrong(nn.Module):
    def __init__(self,in_feature,out_feature=1):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(in_feature, 128))
        self.dense2 = nn.Sequential(nn.Linear(128, 64))
        self.classif = nn.Sequential(nn.Linear(64, 1))
        self.sigm=nn.Sigmoid()
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        x=self.dense1(x)
        x=self.dropout(x)
        x=self.dense2(x)
        x=self.dropout(x)
        x=self.classif(x)
        x=self.sigm(x)
        return x
class Linear_probingBase(nn.Module):
    def __init__(self,in_feature,out_feature=1):
        super(Linear_probingBase, self).__init__()
        self.linear_probing=nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.linear_probing(x)
        return x
import torch 
class baseLine(nn.Module):
    def __init__(self,name_model="dinov2_vits14",hub="facebookresearch/dinov2",linear_probing:str="base",device="cuda"):
        super().__init__()

        print("loading the model",name_model)

        self.feature_extractor= torch.hub.load(hub, name_model).to(device)
        #! Important specify the name
        self.feature_extractor.name=name_model
        self.feature_extractor.eval()

        # Determine the number of output features
        self.num_features = self._get_num_features(name_model, self.feature_extractor)
        #! Adapt the model to the new task

        self.adapt_feature_extractor(name_model,self.feature_extractor)

        self.linear_probing=self._select_linear_probing(linear_probing).to(device)
        self.linear_probing=torch.compile(self.linear_probing)
    def _select_linear_probing(self,linear_probing:str="base"):
        if linear_probing=="base":
            return Linear_probingBase(self.num_features)
        elif linear_probing=="strong":
            return Linear_probingStrong(self.num_features)
        else:
            raise ValueError(f"Unknown linear probing type: {linear_probing}")
    def forward(self,x):
        x=self.feature_extractor(x)
        x=self.linear_probing(x)
        return x
    def adapt_feature_extractor(self,name_model,model):
        #! Change the last head or the last layer of the model to adapt it to the new task

        if name_model.startswith("vgg"): 
            model.classifier[6] = torch.nn.Identity()
        elif name_model.startswith("resnet"):
            model.fc= torch.nn.Identity()
        elif name_model.startswith("mobilenet"):
            model.classifier=torch.nn.Identity()
        elif name_model.startswith("swin"):
            model.head=torch.nn.Identity()
        elif "dino" in name_model:
            pass 
        else:   
            raise ValueError(f"Unknown model architecture: {name_model}")
        return 
    def _get_num_features(self, name_model, model):
        # Map model names to the output feature dimension of their penultimate layer
        if name_model.startswith("resnet"): #changer  fc 
            return model.fc.in_features
        elif name_model.startswith("vgg"): 
            return model.classifier[6].in_features #  changer le classifier 6 pour revenir au hidden model
        elif name_model.startswith("mobilenet"): #  Classifier 3
            return model.classifier[0].in_features
        elif name_model.startswith("swin"): # changer head 
            return model.head.in_features

        elif "dino" in name_model:
            return model.num_features
        else:
            raise ValueError(f"Unknown model architecture: {name_model}")