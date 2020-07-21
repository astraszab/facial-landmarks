import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def train_net(net, trainloader, valloader, criterion, optimizer, device,
              num_epochs=10, visualize=True, lr_scheduler=None, eval_period=2000):
    available_schedulers = (
        optim.lr_scheduler.ReduceLROnPlateau,
        optim.lr_scheduler.CyclicLR,
        optim.lr_scheduler.OneCycleLR,
    )
    if lr_scheduler is not None and not isinstance(lr_scheduler, available_schedulers):
        print('Warning. Current lr_scheduler may work not in a proper way.')
    net.to(device)
    net.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'].to(device), data['landmarks'].to(device)
            labels = labels.reshape(labels.shape[0], -1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                if not isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step()
            running_loss += loss.item()
            if i % eval_period == eval_period - 1:
                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_data in valloader:
                        images, labels = val_data['image'].to(device), val_data['landmarks'].to(device)
                        labels = labels.reshape(labels.shape[0], -1)
                        outputs = net(images)
                        val_loss += criterion(outputs, labels).item()
                val_loss /= len(valloader)
                val_losses.append(val_loss)
                running_loss /= eval_period
                train_losses.append(running_loss)
                print('[%d, %5d] train_loss: %.3f, val_loss: %.3f' %
                      (epoch + 1, i + 1, running_loss, val_loss))
                running_loss = 0.0
                net.train()
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_losses[-1])
    print('Finished training')
    if visualize:
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend()
        plt.show()
