# CNN-VGG-for-CIFAR
The 2nd project of the course Neural Network and Deep Learning, Fudan Univ.


## Project Summary
This is a CIFAR-10 based deep learning experiment project designed to compare different model architectures, optimizers, and batch normalization effects on training performance. The project contains two main components:

- `main.py`: trains and evaluates models, generates training loss and accuracy curves, comparison plots, logs, and saves model weights.
- `VGG_landscape.py`: compares the loss landscapes and gradient behavior of `VGG_A` and `VGG_A_BatchNorm`, analyzing how batch normalization affects training stability and loss geometry.

## Directory Structure

- `main.py`: main training entry point.
- `VGG_landscape.py`: VGG loss landscape comparison script.
- `data/`: data loading utilities, including the CIFAR-10 data loader.
- `models/`: model definitions, including VGG, ResNet, and TinyResNet.
- `utils/`: training, validation, recording, and plotting utilities.
- `outputs/`: output directory for training results.

## main.py Training and Evaluation

`main.py` includes a `runner` class that handles the full training workflow:

1. Select model:
   - `resnet` -> `CIFARResNet`
   - `tinyresnet` -> `TinyResNet`
   - `vgg_a` -> `VGG_A`
   - `vgg_a_bn` -> `VGG_A_BatchNorm`
2. Select optimizer:
   - `sgd`: with `momentum=0.9`, Nesterov acceleration, and weight decay.
   - `adamw`: with weight decay.
3. Data loading: Uses `get_cifar_loader()` from `data/loaders.py` to load CIFAR-10. The training dataset applies random crop, horizontal flip, color jitter, and normalization.
4. Loss function: `CrossEntropyLoss(label_smoothing=0.1)`.
5. Additional training features:
   - MixUp data augmentation controlled by `mixup_alpha`.
   - Learning rate scheduling: linear warmup for `warmup_epochs=5`, followed by `CosineAnnealingLR`.
   - Gradient clipping: `clip_grad_norm_(max_norm=1.0)`.
6. After training, validation and result recording are performed:
   - `utils/recorder.py` saves loss plots, accuracy plots, training logs, and model weights.

### Default Training Configuration

The default settings in `main.py` are:

- batch size: `256`
- epochs: `100`
- weight decay: `5e-4`
- mixup alpha: `1.0`
- resnet + sgd: learning rate `0.02`
- resnet + adamw: learning rate `0.01`
- tinyresnet + sgd: learning rate `0.02`
- tinyresnet + adamw: learning rate `0.01`

### Training Outputs

After training, the project generates:

- `outputs/lines/`: loss and accuracy curve plots for each experiment.
- `outputs/comparisons/`: comparison plots for multiple experiments.
- `outputs/logs/`: log files containing training parameters, test accuracy, and elapsed time.
- `outputs/saved_models/`: saved model weight files.

## VGG_landscape.py Comparison

`VGG_landscape.py` analyzes the training loss curves and gradient behavior of `VGG_A` and `VGG_A_BatchNorm` across different learning rates. The main workflow is:

1. Call `record_landscapes(model, lrs, params_base, op_n)`: train the model for each learning rate and record the loss curve.
2. Compute the min/max envelope across learning rate curves.
3. Plot and save comparison figures for the VGG loss landscape, gradient change, and maximum gradient difference.

### Key Output Files

The `outputs/vgg-results/` directory will contain:

- `vgg_landscapes2.png`: loss envelope comparison between `VGG_A` and `VGG_A_BatchNorm`.
- `vgg_grad_change.png`: gradient change comparison derived from the loss curves.
- `vgg_max_grad_diff.png`: maximum gradient difference over step distances comparison.

### Analysis Goals

This script is primarily used to compare:

- Whether batch normalization stabilizes VGG loss curves.
- How model loss landscapes vary across different learning rates.
- The relationship between gradient change magnitude and training step distances.

## Dependencies

Recommended Python libraries:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

## Running the Project

```bash
python main.py
python VGG_landscape.py
```

If you want to run only a specific model or optimizer, modify the `r.run(...)` calls in `main.py` accordingly.

## Model Descriptions

The project includes the following available models:

- `resnet`: CIFARResNet base ResNet architecture.
- `tinyresnet`: a compact ResNet variant.
- `vgg_a`: the classic VGG-A model.
- `vgg_a_bn`: VGG_A with Batch Normalization added.

---

To extend this project, add more models, optimizers, or training strategies in `main.py`, or add additional loss landscape analysis metrics in `VGG_landscape.py`.
