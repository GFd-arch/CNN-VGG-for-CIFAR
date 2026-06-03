import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt

from main import runner


def record_landscapes(model, lrs=None, params_base=None, op_n='adamw', max_step=None):
    if lrs is None:
        lrs = [1e-3, 2e-3, 1e-4, 5e-4]
    if params_base is None:
        params_base = [128, lrs[0], 100, 5e-4, 0.0]

    all_curves = []
    for lr in lrs:
        params = list(params_base)
        params[1] = lr
        r = runner()
        _, losses, _ = r.run(model_n=model, op_n=op_n, params=params)
        if max_step is not None:
            losses = losses[:max_step]
        all_curves.append(losses)

    # Ensure all curves have the same length by truncating to shortest
    if len(all_curves) == 0:
        return [], []

    n = min(len(c) for c in all_curves)
    min_curve, max_curve = [], []
    for i in range(n):
        vals = [curve[i] for curve in all_curves]
        min_curve.append(min(vals))
        max_curve.append(max(vals))

    return min_curve, max_curve


def save_landscapes_plot(curves_dict, save_path):
    dir = os.path.dirname(save_path)
    if dir:
        os.makedirs(dir, exist_ok=True)

    plt.figure()
    for label, (min_curve, max_curve) in curves_dict.items():
        if not min_curve or not max_curve:
            continue
        x = range(len(min_curve))
        plt.fill_between(x, min_curve, max_curve, alpha=0.5, label=label)

    plt.xlabel('Iteration Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def gradient_predictiveness(loss_curve):
    import numpy as _np

    if loss_curve is None or len(loss_curve) == 0:
        return _np.array([]), _np.array([])

    losses = _np.array(loss_curve)
    grads = _np.gradient(losses)
    grad_changes = _np.abs(_np.diff(grads))
    return grads, grad_changes


def max_gradient_diff_over_distance(loss_curve, max_distance=10):
    import numpy as _np

    if loss_curve is None or len(loss_curve) == 0:
        return {}

    losses = _np.array(loss_curve)
    grads = _np.gradient(losses)
    results = {}
    L = len(grads)
    for d in range(1, min(max_distance, L - 1) + 1):
        diffs = _np.abs(grads[d:] - grads[:-d])
        results[d] = float(diffs.max()) if diffs.size > 0 else 0.0
    return results


def save_gradient_change_plot(curves_dict, save_path):
    import numpy as _np

    dir = os.path.dirname(save_path)
    if dir:
        os.makedirs(dir, exist_ok=True)

    plt.figure()
    if not curves_dict:
        plt.text(0.5, 0.5, 'no data', ha='center')
    else:
        for label, grad_changes in curves_dict.items():
            if grad_changes is None or len(grad_changes) == 0:
                continue
            x = range(len(grad_changes))
            plt.plot(x, _np.array(grad_changes), label=label)
    plt.xlabel('Iteration Step')
    plt.ylabel('Absolute change in gradient')
    if curves_dict:
        plt.legend()
    plt.savefig(save_path)
    plt.close()


def save_max_gradient_diff_plot(results_dict, save_path):
    dir = os.path.dirname(save_path)
    if dir:
        os.makedirs(dir, exist_ok=True)

    plt.figure()
    if not results_dict:
        plt.text(0.5, 0.5, 'no data', ha='center')
    else:
        for label, model_results in results_dict.items():
            if not model_results:
                continue
            distances = sorted(model_results.keys())
            values = [model_results[d] for d in distances]
            plt.plot(distances, values, marker='o', label=label)
    plt.xlabel('Distance (steps)')
    plt.ylabel('Max abs gradient difference')
    if results_dict:
        plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    dir = 'outputs/vgg-results'
    os.makedirs(dir, exist_ok=True)

    lrs = [1e-3, 2e-3, 1e-4, 5e-4]
    params_base = [128, lrs[0], 100, 5e-4, 0.0]
    max_step = None  # change to an int to limit recorded steps per run

    vgga_lines = record_landscapes('vgg_a', lrs=lrs, params_base=params_base, op_n='adamw', max_step=max_step)
    vggabn_lines = record_landscapes('vgg_a_bn', lrs=lrs, params_base=params_base, op_n='adamw', max_step=max_step)

    save_landscapes_plot({'VGG_A': vgga_lines, 'VGG_A_BN': vggabn_lines}, f'{dir}/vgg_landscapes2.png')

    # Compute and save gradient predictiveness (change of gradient) for each model's aggregated curve
    # Use the midpoint curve (mean of min/max) for the metric
    import numpy as _np

    def midpoint_curve(pair):
        if not pair or not pair[0] or not pair[1]:
            return []
        a = _np.array(pair[0])
        b = _np.array(pair[1])
        n = min(len(a), len(b))
        return ((a[:n] + b[:n]) / 2.0).tolist()

    vgga_mid = midpoint_curve(vgga_lines)
    vggabn_mid = midpoint_curve(vggabn_lines)

    _, vgga_grad_changes = gradient_predictiveness(vgga_mid)
    _, vggabn_grad_changes = gradient_predictiveness(vggabn_mid)

    save_gradient_change_plot(
        {'VGG_A': vgga_grad_changes, 'VGG_A_BN': vggabn_grad_changes},
        f'{dir}/vgg_grad_change.png'
    )

    # Compute and save maximum gradient difference over distances
    maxd = 20
    vgga_maxdiff = max_gradient_diff_over_distance(vgga_mid, max_distance=maxd)
    vggabn_maxdiff = max_gradient_diff_over_distance(vggabn_mid, max_distance=maxd)

    save_max_gradient_diff_plot(
        {'VGG_A': vgga_maxdiff, 'VGG_A_BN': vggabn_maxdiff},
        f'{dir}/vgg_max_grad_diff.png'
    )

