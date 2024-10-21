import torch


def compute_accuracy(model, dataloader, device):
    """
    Computes the accuracy of the model on the given dataloader.

    Args:
        model (torch.nn.Module): model to evaluate
        dataloader (torch.utils.data.DataLoader): dataloader to evaluate the model
        device (torch.device): device to use
    Returns:
        float: accuracy
    """

    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    correct_top1 = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Calculate top-1 accuracy
            _, predicted = outputs.topk(1, dim=1)
            correct_top1 += predicted.eq(labels.view(-1, 1)).sum().item()

            total_samples += labels.size(0)

    accuracy_top1 = correct_top1 / total_samples
    del (
        labels,
        images,
        total_samples,
        correct_top1,
        outputs,
        predicted,
    )
    return accuracy_top1
