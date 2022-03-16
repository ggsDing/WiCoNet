import torch
import torch.nn.functional as F
import torch.nn as nn
# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=0,):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
    
def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    
    if weight_pos is not None:        
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def pix_loss(output, target, pix_weight, ignore_index=None):
    # Calculate log probabilities
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos
        
    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Multiply with weights
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    # Average over mini-batch
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss
    