import torch
import torch.nn as nn
import numpy as np 
import math
import pandas as pd

#####################################
### MLP

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_fn=nn.Tanh()):
        super(MLP, self).__init__()
        self.in_dim = input_size
        self.out_dim = output_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn

        self.in_layer = nn.Linear(self.in_dim, self.hidden_units)
        self.hidden = nn.ModuleList(
            [nn.Linear(self.hidden_units, self.hidden_units) for _ in range(self.hidden_layers)]
        )
        self.out_layer = nn.Linear(self.hidden_units, self.out_dim)

    def forward(self, x, y, z, t):
        x = torch.cat([x, y, z, t], dim=1)
        x = self.in_layer(x)
        x = self.activation_fn(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.out_layer(x)
        return x




###################################
### MLP Free divergence

class DivergenceFreeMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        hidden_units,
        sigma_x,
        sigma_y,
        sigma_z,
        activation_fn=nn.Tanh(),
    ):
        """
        Red libre de divergencia basada en potencial vectorial.

        La red interna predice:

            A = (A_x, A_y, A_z)

        y la velocidad se define como:

            u = curl(A)

        Entradas:
            x, y, z, t normalizadas.

        Salidas:
            u, v, w adimensionales.

        Importante:
            sigma_x, sigma_y, sigma_z corresponden a la normalización
            de las coordenadas adimensionales x', y', z'.
        """

        super(DivergenceFreeMLP, self).__init__()

        self.sigma_x = float(np.asarray(sigma_x).squeeze())
        self.sigma_y = float(np.asarray(sigma_y).squeeze())
        self.sigma_z = float(np.asarray(sigma_z).squeeze())

        # Red que predice el potencial vectorial A = (A_x, A_y, A_z)
        self.potential_net = MLP(
            input_size=input_size,
            output_size=3,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation_fn=activation_fn,
        )

    def grad(self, f, x):
        """
        Derivada df/dx usando autograd.
        """

        return torch.autograd.grad(
            f,
            x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

    def forward(self, x, y, z, t):
        """
        Retorna velocidad divergence-free:

            [u, v, w] = curl(A)
        """

        # Necesario porque la salida depende de derivadas respecto a x,y,z.
        if not x.requires_grad:
            x.requires_grad_(True)
        if not y.requires_grad:
            y.requires_grad_(True)
        if not z.requires_grad:
            z.requires_grad_(True)
        if not t.requires_grad:
            t.requires_grad_(True)

        A = self.potential_net(x, y, z, t)

        A_x = A[:, 0:1]
        A_y = A[:, 1:2]
        A_z = A[:, 2:3]

        # ========================================================
        # Derivadas respecto a coordenadas normalizadas
        # ========================================================

        A_x_yhat = self.grad(A_x, y)
        A_x_zhat = self.grad(A_x, z)

        A_y_xhat = self.grad(A_y, x)
        A_y_zhat = self.grad(A_y, z)

        A_z_xhat = self.grad(A_z, x)
        A_z_yhat = self.grad(A_z, y)

        # ========================================================
        # Corrección por normalización
        # d/dx' = (1/sigma_x) d/dx_hat
        # ========================================================

        A_x_y = A_x_yhat / self.sigma_y
        A_x_z = A_x_zhat / self.sigma_z

        A_y_x = A_y_xhat / self.sigma_x
        A_y_z = A_y_zhat / self.sigma_z

        A_z_x = A_z_xhat / self.sigma_x
        A_z_y = A_z_yhat / self.sigma_y

        # ========================================================
        # Velocidad como rotacional del potencial vectorial
        # ========================================================
        # u = dA_z/dy - dA_y/dz
        # v = dA_x/dz - dA_z/dx
        # w = dA_y/dx - dA_x/dy

        u = A_z_y - A_y_z
        v = A_x_z - A_z_x
        w = A_y_x - A_x_y

        out = torch.cat([u, v, w], dim=1)

        return out

    def potential(self, x, y, z, t):
        """
        Retorna el potencial vectorial A = (A_x,A_y,A_z).
        Esto es útil para debug, pero no se usa directamente en la pérdida.
        """

        return self.potential_net(x, y, z, t)