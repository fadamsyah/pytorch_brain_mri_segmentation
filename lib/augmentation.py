import numpy as np

from PIL import Image
from torchvision import transforms

class Scale(object):
    def __init__(self, scale, constant=0.):
        self.scale = scale
        self.constant = constant
        
    def __call__(self, sample):
        image, mask = sample
               
        scale = np.random.uniform(low = -self.scale, high = self.scale)
        image = Scale.scaling(image, scale, self.constant)
        mask = Scale.scaling(mask, scale, self.constant)
        
        return image, mask
    
    @staticmethod
    def scaling(image, scale, constant):
        w, h = image.size
        
        nw = int((1. + scale) * w + 0.5)
        nh = int((1. + scale) * h + 0.5)
        
        if scale == 0: return image
        
        # Resize the image
        image = image.resize((nw, nh), Image.BILINEAR)
        
        if scale > 0.:
            # Center crop the image    
            
            left = (nw - w) // 2
            top = (nh - h) // 2
            right = w + left
            bottom = h + top
            
            image = image.crop((left, top, right, bottom))
        
        elif scale < 0:
            # Pad the image with constant
            
            left = (w - nw) // 2
            top = (h - nh) // 2
            right = w - nw - left
            bottom = h - nh - top
            
            padding = ((top, bottom), (left, right), (0,0))
            
            image = np.asarray(image)
            
            # We have to pick different padding because of the array shape
            # i.e. here, for 3 channels we have (W x H x 3)
            # but, for 1 channel we have (W x H)
            if len(image.shape) < 3: padding = ((top, bottom), (left, right))
            else: padding = ((top, bottom), (left, right), (0,0))
            
            image = np.pad(image, padding,
                           mode="constant",
                           constant_values=constant)
            image = Image.fromarray(image)
            
        return image

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle # degree
        
    def __call__(self, sample):
        image, mask = sample
        
        angle = np.random.uniform(low = -self.angle, high = self.angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        
        return image, mask
    
class HorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, sample):
        image, mask = sample
        
        if self.prob >= np.random.uniform(low=0., high=1.):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        return image, mask

class VerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, sample):
        image, mask = sample
        
        if self.prob >= np.random.uniform(low=0., high=1.):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image, mask
    
def SegmentationTransforms(scale=None, angle=None,
                           hflip_prob=None, vflip_prob=None,
                           sequence=['scale', 'rotate',
                                     'horizontal_flip',
                                     'vertical_flip']):
    transform_list = []
    
    for seq in sequence:
        if scale and (seq.lower() == 'scale'): transform_list.append(Scale(scale))
        if angle and (seq.lower() == 'rotate'): transform_list.append(Rotate(angle))
        if hflip_prob and (seq.lower() == 'horizontal_flip'): transform_list.append(HorizontalFlip(hflip_prob))
        if vflip_prob and (seq.lower() == 'vertical_flip'): transform_list.append(VerticalFlip(vflip_prob))
    
    return transforms.Compose(transform_list)