import numpy as np
import torch
import time
import copy

def train_model(device, model, dataloaders, dataset_sizes,
                criterion, optimizer, scheduler=None,
                metric=None, num_epochs=25):
    since = time.time()
    
    history_loss = {'train': [], 'val': []}
    history_metric = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    if metric:
        if metric.target.lower() == 'max': best_metric = - np.Inf
        elif metric.target.lower() == 'min': best_metric = np.Inf

    for epoch in range(num_epochs):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            if metric: running_metric = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if metric: running_metric += metric(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)
                    
            if phase == 'train':
                if scheduler: scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            if metric: epoch_metric = running_metric / dataset_sizes[phase]
            
            history_loss[phase].append(epoch_loss)
            if metric:
                print('{} Loss: {:.4f} {}: {:.4f}'.format(
                    phase, epoch_loss, metric.name, epoch_metric))
                history_metric[phase].append(epoch_metric)
            else:
                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

            # deep copy the model
            if (metric is not None) and (phase == 'val'):
                
                if (metric.target.lower() == 'max') and (epoch_metric > best_metric):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                elif (metric.target.lower() == 'min') and (epoch_metric < best_metric):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if metric:
        print('Best val {}: {:4f}'.format(metric.name, best_metric))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return {'model': model, 'history_loss': history_loss, 'history_metric': history_metric}