import numpy as np
import matplotlib.pyplot as plt
import torch

from .metrics import dice_score

def cfirst_to_clast(images):
    """
    Args:
        images: numpy array (N, C, W, H) or (C, W, H)
    return:
        images: numpy array(N, W, H, C) or (W, H, C)
    """
    images = np.swapaxes(images, -3, -2)
    images = np.swapaxes(images, -2, -1)
    return images

def visualize(model, images, masks, threshold=0.5, figsize=(16, 26)):
    """
    Args:
        model: torch model
        images: numpy array or tensor (N, 3, W, H)
        masks: numpy array or tensor (N, 1, W, H)
        threshold: float
    """
    
    model.eval()
    with torch.set_grad_enabled(False):
        inputs = images.to(device, dtype=torch.float)
        outputs = model(inputs)
        
    outputs = outputs.detach().cpu().numpy()
    outputs[outputs >= threshold] = 1.
    outputs[outputs <= threshold] = 0.
    
    images = np.array(images)
    masks = np.array(masks)
    
    dsc = list(map(dice_score, masks, outputs))
    
    images = cfirst_to_clast(images)
    masks = cfirst_to_clast(masks)
    outputs = cfirst_to_clast(outputs)
    
    num_images = masks.shape[0]
    
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(num_images, 3, 1 + 3 * i)
        plt.imshow(images[i])
        plt.xlabel(f'{dsc[i]}')
        
        plt.subplot(num_images, 3, 2 + 3 * i)
        plt.imshow(outputs[i])
        plt.xlabel('prediction')
        
        plt.subplot(num_images, 3, 3 + 3 * i)
        plt.imshow(masks[i])
        plt.xlabel('ground-truth')
    plt.show()
