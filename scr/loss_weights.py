import torch
import numpy as np
from loss_functions import data_loss, pressure_gradient_loss



def compute_grad_norms(model, loss_fn):

    model.zero_grad(set_to_none=True)

    loss = loss_fn()
    loss.backward()

    all_grads = []

    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(
                param.grad.detach().reshape(-1)
            )

    if len(all_grads) == 0:
        grad_norm = 0.0

    else:
        all_grads = torch.cat(all_grads)

        grad_norm = all_grads.norm(
            p=2
        ).item()

    model.zero_grad(set_to_none=True)

    return grad_norm


############################################################
### Data weights update function

def update_loss_weights_data(
    model,
    loss_weights,
    X_data_batch_tensor,
    alpha=0.9,
    lambda_min=1e-3,
    lambda_max=1e3
):

    eps = 1e-12

    # ============================================================
    # Funciones locales de pérdida
    # ============================================================

    def loss_u_fn():
        X = X_data_batch_tensor.detach().clone().requires_grad_(True)

        loss_u, loss_v, loss_w = data_loss(
            model,
            X,
        )

        return loss_u


    def loss_v_fn():
        X = X_data_batch_tensor.detach().clone().requires_grad_(True)

        loss_u, loss_v, loss_w = data_loss(
            model,
            X,
        )

        return loss_v


    def loss_w_fn():
        X = X_data_batch_tensor.detach().clone().requires_grad_(True)

        loss_u, loss_v, loss_w = data_loss(
            model,
            X,
        )

        return loss_w


    # ============================================================
    # Normas de gradiente
    # ============================================================

    grad_norm_u = compute_grad_norms(model, loss_u_fn)
    grad_norm_v = compute_grad_norms(model, loss_v_fn)
    grad_norm_w = compute_grad_norms(model, loss_w_fn)

    grad_norms = np.array(
        [
            grad_norm_u,
            grad_norm_v,
            grad_norm_w,
        ],
        dtype=np.float64,
    )

    # ============================================================
    # Pesos nuevos
    # ============================================================

    grad_mean = np.mean(grad_norms)

    weights_new = grad_mean / (grad_norms + eps)

    weights_new = np.clip(
        weights_new,
        lambda_min,
        lambda_max,
    )

    # Mantener la escala media de las perdidas de datos en uno.
    weights_new = weights_new / max(np.mean(weights_new), eps)

    # ============================================================
    # Suavizado exponencial
    # ============================================================

    loss_weights_new = (
        alpha * loss_weights
        + (1.0 - alpha) * weights_new
    )

    return loss_weights_new, grad_norms


def update_loss_weights_pressure(
    model,
    loss_weights,
    X_batch_tensor,
    Y_grad_batch_tensor,
    sigma_x,
    sigma_y,
    sigma_z,
    alpha=0.9,
    lambda_min=1e-3,
    lambda_max=1e3,
):
    """Balancea acc, adv y vis por sus normas de gradiente."""

    eps = 1e-12

    def component_loss(index):
        def loss_fn():
            X = X_batch_tensor.detach().clone().requires_grad_(True)
            losses = pressure_gradient_loss(
                model=model,
                X_tensor_norm=X,
                Y_grad_tensor=Y_grad_batch_tensor,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                sigma_z=sigma_z,
            )
            return losses[index]

        return loss_fn

    grad_norms = np.array(
        [
            compute_grad_norms(model, component_loss(0)),
            compute_grad_norms(model, component_loss(1)),
            compute_grad_norms(model, component_loss(2)),
        ],
        dtype=np.float64,
    )

    grad_mean = np.mean(grad_norms)
    weights_new = grad_mean / (grad_norms + eps)
    weights_new = np.clip(weights_new, lambda_min, lambda_max)
    weights_new = weights_new / max(np.mean(weights_new), eps)

    loss_weights_new = (
        alpha * loss_weights
        + (1.0 - alpha) * weights_new
    )

    return loss_weights_new, grad_norms

