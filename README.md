# Brain MRI Segmentation with UNet on PyTorch

## References

The references used in this repository are:
- https://github.com/mateuszbuda/brain-segmentation-pytorch
- https://www.kaggle.com/anantgupt/brain-mri-detection-segmentation-resunet
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## Notebook

https://www.kaggle.com/fadillahadamsyah/unet-for-brain-mri-segmentation

## Reusable Files

On the `lib` folder, you can find the following files:
- `augmentation.py` Custom augmentation for image segmentation training, i.e.:
    - scale
    - rotate
    - horizontal flip
    - vertical flip
- `dataset.py` Custom dataset class for image segmentation. The pandas and PIL are used to implement the class. The paths of images are located on the dataframe.
- `loss.py` Custom Dice Loss implementation for segmentation training.
- `metrics.py` Custom Dice Score implementation for segmentation metric.
- `model.py` UNet wrapper
- `train.py` Custom training loop. The output is a dictionary containing *'model'*, *'history_loss'*, and *'history_metric'* keys. The arguments are:
    - device
    - model
    - dataloaders (a dictionary with key *'train'* and *'val'*)
    - dataset_sizes (a dictionary with key *'train'* and *'val'*)
    - criterion
    - optimizer
    - scheduler (optional)
    - metric (optional, class)
    - num_epochs
- `visualization.py` Plot samples result.