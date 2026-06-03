import torch
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion,
                 optimizer, scheduler, device, epochs=10, print_freq=100, mixup_alpha=1.0):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.print_freq = print_freq
        self.mixup_alpha = mixup_alpha
        self.losses = []
        self.acc = []


    def train(self):
        print("Training...")
        self.model.to(self.device)            

        for e in range(self.epochs):
            self.model.train()
            # Loss = 0
            for i, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                mixed_X, y_a, y_b, lam = self.mixup(X, y, self.mixup_alpha)

                self.optimizer.zero_grad()
                y_pred = self.model(mixed_X)
                loss = lam * self.criterion(y_pred, y_a) + (1 - lam) * self.criterion(y_pred, y_b)
                loss.backward()
                # Loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.losses.append(loss.item())

                if (i+1) % self.print_freq == 0:
                    print(f'Epoch [{e+1}/{self.epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

            # self.losses.append(Loss / min(len(self.train_loader), 1))

            if self.scheduler is not None:
                # Use scheduler.step() without an explicit epoch argument to avoid PyTorch deprecation warnings.
                self.scheduler.step()
            
            if (e+1) % 10 == 0 or e == self.epochs - 1:
                self.eval()
                

    def eval(self):
        print("Evaluating on test set...")
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        self.acc.append(correct / total)

    
    def mixup(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam



class Validator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.datas = test_loader
        self.device = device
        self.acc = None

    def accuracy(self):
        print("Evaluating on validation set...")
        self.model.eval()

        with torch.no_grad():
            corr, total = 0, 0
            for X, y in self.datas:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                corr += (predicted == y).sum().item()

        self.acc = corr / total
        return self.acc

