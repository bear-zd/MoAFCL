import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
import math

class DecomposedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,bias: bool = True) -> None:
        super(DecomposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sw = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.Tensor(out_features))
        self.aw = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.sw, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.sw)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.mask, -bound, bound)
    def set_atten(self,t,dim):
        if t==0:
            self.atten = Parameter(torch.zeros(dim))
            self.atten.requires_grad=False
        else:
            self.atten = Parameter(torch.rand(dim))
    def set_knlwledge(self,from_kb):
        self.from_kb = from_kb

    def get_weight(self):
        m = nn.Sigmoid()
        sw = self.sw.transpose(0, -1)

        weight = (sw * m(self.mask)).transpose(0, -1) + self.aw + torch.sum(self.atten * self.from_kb.cuda(), dim=-1)
        weight = weight.type(torch.cuda.FloatTensor)
        return weight
    def forward(self, input):
        weight = self.get_weight()
        return F.linear(input, weight, self.bias)
