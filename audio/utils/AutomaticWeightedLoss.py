import torch
import torch.nn as nn
import pdb
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num):
        super(AutomaticWeightedLoss, self).__init__()
        self.num = num
        log_var = torch.zeros(num).cuda()
        self.log_var = torch.nn.Parameter(log_var)
        self.var = torch.exp(-self.log_var)

    def forward(self, loss_dict):
        weight_loss = dict()
        assert len(loss_dict) == self.num
        loss_sum = 0
        for i, loss in enumerate(loss_dict.keys()):
            weight_loss[loss] = self.var[i] * loss_dict[loss]
        #pdb.set_trace()
        loss_sum = sum(weight_loss.values()) + self.log_var.sum()
        return loss_sum, weight_loss



class Uncertainty_weighed_loss(nn.Module):
    def __init__(self):
        super(Uncertainty_weighed_loss, self).__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)
        self.weight = torch.exp(-self.log_var)
    def forward(self, loss):
        loss = loss * self.weight + self.log_var
        return loss,self.log_var


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())


