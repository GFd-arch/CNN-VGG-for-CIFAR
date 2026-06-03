import torch
import matplotlib.pyplot as plt
import os


def plot_loss(losses, acc, types=['resnet', 'sgd', 'False'], title='Training Loss', dir='outputs/lines'):

    os.makedirs(dir, exist_ok=True)
    model_n, op_n, mixup = types

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.title(f'{title} - {model_n} - {op_n} - {bool(mixup)}')
    plt.xlabel(f'Iteration - Accuracy:{acc:.4f}')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir}/{title.replace(" ", "_")}_{model_n}_{op_n}_{bool(mixup)}.png')
    plt.close()


def plot_acc(acc, test_acc, types=['resnet', 'sgd', 'False'], title='Training Accuracy', dir='outputs/lines'):
    os.makedirs(dir, exist_ok=True)
    model_n, op_n, mixup = types

    plt.figure(figsize=(10, 5))
    plt.plot(acc, label='Train Accuracy')
    plt.axhline(y=test_acc, color='r', linestyle='--', label='Test Accuracy')
    plt.title(f'{title} - {model_n} - {op_n} - {bool(mixup)}')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir}/{title.replace(" ", "_")}_{model_n}_{op_n}_{bool(mixup)}.png')
    plt.close()


def plot_comparison(curve_sets, title='Comparison Plot', ylabel='Value', dir='outputs/comparisons'):
    os.makedirs(dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for values, label in curve_sets:
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    safe_title = title.replace(' ', '_')
    label_suffix = '_'.join([label.replace(' ', '_') for _, label in curve_sets])
    plt.savefig(f'{dir}/{safe_title}_{label_suffix}.png')
    plt.close()


def record_logs(params, results, types=['resnet', 'sgd', 'False'], time=0.0, dir='outputs/logs'):

    os.makedirs(dir, exist_ok=True)
    bs, lr, e, wd, mixup = params
    model_n, op_n, mixup = types

    log_file = os.path.join(dir, f'train_log - {model_n} - {op_n} - {bool(mixup)}.txt')
    with open(log_file, 'a') as f:
        f.write(f"Parameters: \n")
        f.write(f"Batch Size: {bs}, Learning Rate: {lr}, Epochs: {e}, Weight Decay: {wd}, Mixup-alpha: {bool(mixup)}")
        f.write(f"\n\nResults:\n")
        f.write(f"{results}\n")
        f.write(f"Training Time: {time:.2f} seconds\n\n\n")
        
    
def save_model(model, types=['resnet, sgd', 'False'], base_dir='outputs/saved_models'):

    os.makedirs(base_dir, exist_ok=True)
    model_n, op_n, mixup = types
    torch.save(model.state_dict(), f'{base_dir}/{model_n}_{op_n}_mixup_{mixup}.pth')