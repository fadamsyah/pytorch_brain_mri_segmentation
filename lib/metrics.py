import numpy as np

def dice_score(y_true, y_pred, smoothing=1e-6):
    """
    Args:
        y_true: numpy array with size (W, H, 1) or (1, W, H)
        y_pred: numpy array with size (W, H, 1) or (1, W, H)
        threshold: float
    return:
        dsc: float
    """   
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()

    return (2. * intersection + smoothing) / (union + smoothing)

class DiceScore():
    def __init__(self, threshold=0.5, smoothing=1e-6):
        self.name = 'DSC'
        self.smoothing = smoothing
        self.target = 'max'
        self.threshold = 0.5
        
    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true: numpy array with size (N, W, H, 1) or (N, 1, W, H)
            y_pred: numpy array with size (N, W, H, 1) or (N, 1, W, H)
            threshold: float
        return:
            dsc: float
        """
        
        y_pred[y_pred >= self.threshold] = 1.
        y_pred[y_pred <= self.threshold] = 0.
        
        dscs = np.array(list(map(dice_score, y_true, y_pred, [self.smoothing for _ in range(y_pred.shape[0])])))
        
        return np.mean(dscs)