import torch.nn as nn

class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        
        y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], -1)
        
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1. - dsc