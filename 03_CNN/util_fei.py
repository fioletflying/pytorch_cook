import torch
from torch import nn
import time

def test_model(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.eval()
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            test_loss += loss_fn(pred, targets).item()
            test_correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
    test_loss /= num_batchs
    test_correct /= size

    return test_loss, test_correct
    
def train_model(model, train_dataloader, test_dataloader,creterion, optimizer, device, scheduler=None, epoch_num=10):
    # 初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight)
    model.apply(init_weights)
    print('training on ', device)
    model.to(device)

    train_size = len(train_dataloader.dataset)
    num_batchs = len(train_dataloader)

    for epoch in range(epoch_num):
        since_time = time.time()
        train_loss, train_correct = 0, 0
        model.train()
        
        for batch, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            pred = model(inputs)
            loss = creterion(pred, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
        
        end_time = time.time()
        train_loss /= num_batchs
        train_correct /= train_size
        _,test_correct = test_model(test_dataloader, model, creterion, device)
        print(f"epoch: {epoch+1},\t\
                train_loss: {train_loss:>3f},\t\
                train_correct:{train_correct:>3f},\t\
                test_correct:{test_correct:>3f},\t\
                test time:{(end_time - since_time):>3f}")

    