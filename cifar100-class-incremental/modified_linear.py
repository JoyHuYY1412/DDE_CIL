import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input, num_head=1):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        if num_head>1:
            out=[]
            head_dim = input.size(1)//num_head
            input_list = torch.split(input, head_dim, dim=1)
            input_list = [F.normalize(input_item, p=2,dim=1) for input_item in input_list]
            weight_list = torch.split(self.weight, head_dim, dim=1)
            weight_list = [F.normalize(weight_item, p=2,dim=1) for weight_item in weight_list]
            for n_input, n_weight in zip(input_list, weight_list):
                out.append(F.linear(n_input, n_weight))
            import pdb; pdb.set_trace()
            out = sum(out)
        else:
            out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out

        # if self.sigma is not None:
        #     out = [self.sigma * out_item for out_item in out]
        return out

class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, num_head=1):
        out1 = self.fc1(x, num_head=num_head)
        out2 = self.fc2(x, num_head=num_head)
        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out

class CosineLinear_bi_feat(Module):
    def __init__(self, in_features1, in_features2, out_features, sigma=True):
        super(CosineLinear_bi_feat, self).__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(out_features, in_features1))
        self.weight2 = Parameter(torch.Tensor(out_features, in_features2))

        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input, mask_feat2=False, mean_feat2=None, eval=False):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)
        input1 = F.normalize(input[:,:self.in_features1], p=2,dim=1)
        if mean_feat2 is not None:
            assert mask_feat2
            input2 = F.normalize(mean_feat2, p=2,dim=1)
        else:
            input2 = F.normalize(input[:,self.in_features1:], p=2,dim=1)
        if mask_feat2:
            with torch.no_grad():
                out2 = F.linear(input2, F.normalize(self.weight2, p=2, dim=1)) 
        else:
            out2 = F.linear(input2, F.normalize(self.weight2, p=2, dim=1)) 
        if eval:
            out = F.linear(input1, F.normalize(self.weight1, p=2, dim=1))
        else:
            out = F.linear(input1, F.normalize(self.weight1, p=2, dim=1)) + out2
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear_bi_feat(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features1, in_features2, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear_bi_feat, self).__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear_bi_feat(in_features1, in_features2, out_features1, False)
        self.fc2 = CosineLinear_bi_feat(in_features1, in_features2, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, mask_feat2=False, mean_feat2=None, eval=False):
        out1 = self.fc1(x, mask_feat2=mask_feat2, mean_feat2=mean_feat2, eval=eval)
        out2 = self.fc2(x, mask_feat2=mask_feat2, mean_feat2=mean_feat2, eval=eval)
        out = torch.cat((out1, out2), dim=1) #concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out


class GroupCosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(GroupCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        #w_norm = self.weight.data.norm(dim=1, keepdim=True)
        #w_norm = w_norm.expand_as(self.weight).add_(self.epsilon)
        #x_norm = input.data.norm(dim=1, keepdim=True)
        #x_norm = x_norm.expand_as(input).add_(self.epsilon)
        #w = self.weight.div(w_norm)
        #x = input.div(x_norm)

        ## self.weight [num_classes, num_features]
        # if self.out_features>50:
        #     import pdb; pdb.set_trace()

        weight_norm = torch.norm(self.weight, 2, 1, keepdim=True)  # [num_classes,1]
        with torch.no_grad():
            weight_norm_square = weight_norm*weight_norm
        out = F.linear(F.normalize(input, p=2,dim=1), \
            self.weight/torch.sqrt(torch.mean(weight_norm_square)))

        # ECCV WA
        # weight_norm = torch.norm(self.weight, 2, 1, keepdim=True)  #[num_classes,1]
        # out = F.linear(F.normalize(input, p=2,dim=1), \
        #     self.weight/torch.mean(weight_norm))

        if self.sigma is not None:
            out = self.sigma * out

        # if self.sigma is not None:
        #     out = [self.sigma * out_item for out_item in out]
        return out

class SplitGroupCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitGroupCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = GroupCosineLinear(in_features, out_features1, False)
        self.fc2 = GroupCosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x, num_head=1):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out
