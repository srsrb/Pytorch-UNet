import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

@torch.inference_mode() 
def evaluate_advection(model, loader, device, amp):
    model.eval()
    total_loss = 0
    num_val_batches = len(loader)
    criterion = nn.MSELoss()

    # itérer sur le jeu de validation
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, flows, targets = batch['I1'], batch['flow'], batch['I2']
            
            # Normalisation des images et cibles
            images = (images - images.min()) / (images.max() - images.min())
            targets = (targets - targets.min()) / (targets.max() - targets.min())

            # déplacer les images et cibles sur le bon device
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            flows = flows.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = targets.to(device=device, dtype=torch.float32)

            # concaténer l'image et le flux pour obtenir l'entrée du modèle
            inputs = torch.cat((images, flows), dim=1)

            # prédire l'image cible
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                predictions = model(inputs)

                # calculer la perte MSE entre les images cibles prédites et réelles
                loss = criterion(predictions, targets)
                total_loss += loss.item()

    return total_loss / num_val_batches
