from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from NN import MLP


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Configuración
# ============================================================

experiment_name = "13mm_CFD"
## 9mm
# run_name = "seed_001_20260814_140741"
# run_name = "seed_002_20260814_140850"
# run_name = "seed_003_20260814_140927"
# run_name = "seed_004_20260814_140955"
# run_name = "seed_005_20260814_141029"

## 11mm
# run_name = "seed_001_20260814_141212"
#run_name = "seed_002_20260814_141241"
# run_name = "seed_003_20260814_141305"
# run_name = "seed_004_20260814_141340"
# run_name = "seed_005_20260814_141359"

## 13mm
# run_name = "seed_001_20260816_155204"
# run_name = "seed_002_20260816_155223"
# run_name = "seed_003_20260816_155250"
# run_name = "seed_004_20260816_155312"
run_name = "seed_005_20260816_155339"

## normal 
# run_name = "seed_001_20260814_142008"
# run_name = "seed_002_20260814_142120"
# run_name = "seed_003_20260814_142709"
# run_name = "seed_004_20260814_142813"
# run_name = "seed_005_20260814_142838"


# Si True, además de guardar en derived_terms/,
# copia los archivos a p22/data/ para reutilizar scripts actuales.
# Para una sola semilla está bien. Luego, con 5 semillas,
# será mejor que presión lea desde derived_terms/ del run correspondiente.
COPY_TO_DATA_DIR = True

# Batch size para autograd de 1ra y 2da derivada.
# Si ves problemas de memoria, baja a 1000 o 500.
BATCH_SIZE = 1500


# ============================================================
# Paths
# ============================================================

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

experiment_dir = project_dir / experiment_name
data_dir = experiment_dir / "data"
data_train_dir = experiment_dir / "data_train"

run_dir = data_train_dir / "results" / run_name
checkpoints_dir = run_dir / "checkpoints"
sampling_dir = run_dir / "sampling"
derived_dir = run_dir / "derived_terms"

derived_dir.mkdir(parents=True, exist_ok=True)

best_ckpt_path = checkpoints_dir / "best_velocity.pth"
data_path = data_dir / "centerline_velocity_vectors_13mm_array3d.npy"

if not best_ckpt_path.exists():
    raise FileNotFoundError(f"No existe:\n{best_ckpt_path}")

if not data_path.exists():
    raise FileNotFoundError(f"No existe:\n{data_path}")


# ============================================================
# Helper autograd
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# ============================================================
# Cargar checkpoint y reconstruir red
# ============================================================

checkpoint = torch.load(best_ckpt_path, map_location=device)

# En best_velocity.pth guardamos dic_adim_norm dentro de extra_state
extra_state = checkpoint.get("extra_state", {})

dic_adim_norm = extra_state["dic_adim_norm"]

mu_x = float(dic_adim_norm["mu_x"])
mu_y = float(dic_adim_norm["mu_y"])
mu_z = float(dic_adim_norm["mu_z"])
mu_t = float(dic_adim_norm["mu_t"])

sigma_x = float(dic_adim_norm["sigma_x"])
sigma_y = float(dic_adim_norm["sigma_y"])
sigma_z = float(dic_adim_norm["sigma_z"])
sigma_t = float(dic_adim_norm["sigma_t"])

L = float(extra_state["L"])
T = float(extra_state["T"])
U = float(extra_state["U"])
rho = float(extra_state["rho"])
mu = float(extra_state["mu"])
Re = float(extra_state["Re"])

nu = mu / rho

net = MLP(
    input_size=4,
    output_size=3,
    hidden_layers=4,
    hidden_units=64,
    activation_fn=nn.SiLU(),
).to(device)

net.load_state_dict(checkpoint["model_state_dict"])
net.eval()

print("=" * 90)
print(f"Device:                  {device}")
print(f"Run directory:           {run_dir}")
print(f"Best checkpoint:         {best_ckpt_path}")
print(f"Best epoch:              {checkpoint['epoch']}")
print(f"Best validation metric:  {checkpoint['validation_metric']:.6e}")
print(f"L={L}, T={T}, U={U}, Re={Re:.3f}, nu={nu:.6e}")
print("=" * 90)


# ============================================================
# Cargar data original
# ============================================================

Data = np.load(data_path, allow_pickle=True)   # [Nt, Np, 7]
Nt, Np, ncols = Data.shape

if ncols < 7:
    raise ValueError("Se esperaba un dataset con al menos 7 columnas: x,y,z,t,u,v,w")

Data_flat = Data.reshape(-1, ncols)

coords_xyz_t = Data_flat[:, 0:4]  # físicas: x,y,z,t

# Normalización de entradas
x_phys = coords_xyz_t[:, 0:1]
y_phys = coords_xyz_t[:, 1:2]
z_phys = coords_xyz_t[:, 2:3]
t_phys = coords_xyz_t[:, 3:4]

x_hat = (x_phys / L - mu_x) / sigma_x
y_hat = (y_phys / L - mu_y) / sigma_y
z_hat = (z_phys / L - mu_z) / sigma_z
t_hat = (t_phys / T - mu_t) / sigma_t

X_input = np.concatenate([x_hat, y_hat, z_hat, t_hat], axis=1)

N_total = X_input.shape[0]

print(f"Data shape:              {Data.shape}")
print(f"Flattened points:        {N_total}")
print(f"Batch size:              {BATCH_SIZE}")


# ============================================================
# Buffers de salida
# ============================================================

acc_prime_all = np.zeros((N_total, 3), dtype=np.float64)
adv_prime_all = np.zeros((N_total, 3), dtype=np.float64)
vis_prime_all = np.zeros((N_total, 3), dtype=np.float64)


# ============================================================
# Bucle por batches
# ============================================================

for start in range(0, N_total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N_total)

    Xb = X_input[start:end]

    xb = torch.tensor(Xb[:, 0:1], dtype=torch.float64, device=device, requires_grad=True)
    yb = torch.tensor(Xb[:, 1:2], dtype=torch.float64, device=device, requires_grad=True)
    zb = torch.tensor(Xb[:, 2:3], dtype=torch.float64, device=device, requires_grad=True)
    tb = torch.tensor(Xb[:, 3:4], dtype=torch.float64, device=device, requires_grad=True)

    # Red: entrega u', v', w'
    uvw_prime = net(xb, yb, zb, tb)
    u_p = uvw_prime[:, 0:1]
    v_p = uvw_prime[:, 1:2]
    w_p = uvw_prime[:, 2:3]

    # ========================================================
    # 1) Término acelerativo: ∂u'/∂t'
    # ========================================================

    u_t_hat = grad(u_p, tb)
    v_t_hat = grad(v_p, tb)
    w_t_hat = grad(w_p, tb)

    u_t_prime = u_t_hat / sigma_t
    v_t_prime = v_t_hat / sigma_t
    w_t_prime = w_t_hat / sigma_t

    acc_prime = torch.cat([u_t_prime, v_t_prime, w_t_prime], dim=1)

    # ========================================================
    # 2) Derivadas espaciales de 1er orden: ∂u'/∂x', etc.
    # ========================================================

    u_x_hat = grad(u_p, xb)
    u_y_hat = grad(u_p, yb)
    u_z_hat = grad(u_p, zb)

    v_x_hat = grad(v_p, xb)
    v_y_hat = grad(v_p, yb)
    v_z_hat = grad(v_p, zb)

    w_x_hat = grad(w_p, xb)
    w_y_hat = grad(w_p, yb)
    w_z_hat = grad(w_p, zb)

    u_x_prime = u_x_hat / sigma_x
    u_y_prime = u_y_hat / sigma_y
    u_z_prime = u_z_hat / sigma_z

    v_x_prime = v_x_hat / sigma_x
    v_y_prime = v_y_hat / sigma_y
    v_z_prime = v_z_hat / sigma_z

    w_x_prime = w_x_hat / sigma_x
    w_y_prime = w_y_hat / sigma_y
    w_z_prime = w_z_hat / sigma_z

    # ========================================================
    # 3) Término advectivo: (u' · ∇')u'
    # ========================================================

    adv_u = u_p * u_x_prime + v_p * u_y_prime + w_p * u_z_prime
    adv_v = u_p * v_x_prime + v_p * v_y_prime + w_p * v_z_prime
    adv_w = u_p * w_x_prime + v_p * w_y_prime + w_p * w_z_prime

    adv_prime = torch.cat([adv_u, adv_v, adv_w], dim=1)

    # ========================================================
    # 4) Laplaciano: ∇'^2 u'
    # ========================================================

    u_xx_prime = grad(u_x_hat, xb) / (sigma_x ** 2)
    u_yy_prime = grad(u_y_hat, yb) / (sigma_y ** 2)
    u_zz_prime = grad(u_z_hat, zb) / (sigma_z ** 2)

    v_xx_prime = grad(v_x_hat, xb) / (sigma_x ** 2)
    v_yy_prime = grad(v_y_hat, yb) / (sigma_y ** 2)
    v_zz_prime = grad(v_z_hat, zb) / (sigma_z ** 2)

    w_xx_prime = grad(w_x_hat, xb) / (sigma_x ** 2)
    w_yy_prime = grad(w_y_hat, yb) / (sigma_y ** 2)
    w_zz_prime = grad(w_z_hat, zb) / (sigma_z ** 2)

    lap_u_prime = u_xx_prime + u_yy_prime + u_zz_prime
    lap_v_prime = v_xx_prime + v_yy_prime + v_zz_prime
    lap_w_prime = w_xx_prime + w_yy_prime + w_zz_prime

    vis_prime = (1.0 / Re) * torch.cat(
        [lap_u_prime, lap_v_prime, lap_w_prime],
        dim=1,
    )

    acc_prime_all[start:end] = acc_prime.detach().cpu().numpy()
    adv_prime_all[start:end] = adv_prime.detach().cpu().numpy()
    vis_prime_all[start:end] = vis_prime.detach().cpu().numpy()

    if start % (10 * BATCH_SIZE) == 0 or end == N_total:
        print(f"Processed {end}/{N_total} points")


# ============================================================
# Escala a MKS
# ============================================================

scale_mks = (U ** 2) / L

acc_mks = scale_mks * acc_prime_all
adv_mks = scale_mks * adv_prime_all
vis_mks = scale_mks * vis_prime_all


# ============================================================
# Reensamblar arrays [Nt, Np, 7]
# columnas = x,y,z,t, term_x, term_y, term_z
# ============================================================

Data_acc = np.concatenate([coords_xyz_t, acc_prime_all], axis=1).reshape(Nt, Np, 7)
Data_adv = np.concatenate([coords_xyz_t, adv_prime_all], axis=1).reshape(Nt, Np, 7)
Data_visc = np.concatenate([coords_xyz_t, vis_prime_all], axis=1).reshape(Nt, Np, 7)

Data_acc_mks = np.concatenate([coords_xyz_t, acc_mks], axis=1).reshape(Nt, Np, 7)
Data_adv_mks = np.concatenate([coords_xyz_t, adv_mks], axis=1).reshape(Nt, Np, 7)
Data_visc_mks = np.concatenate([coords_xyz_t, vis_mks], axis=1).reshape(Nt, Np, 7)


# ============================================================
# Guardar en carpeta del run
# ============================================================

np.save(derived_dir / "Data_acc.npy", Data_acc)
np.save(derived_dir / "Data_adv.npy", Data_adv)
np.save(derived_dir / "Data_visc.npy", Data_visc)

np.save(derived_dir / "Data_acc_mks.npy", Data_acc_mks)
np.save(derived_dir / "Data_adv_mks.npy", Data_adv_mks)
np.save(derived_dir / "Data_visc_mks.npy", Data_visc_mks)

print("\nSaved in:")
print(derived_dir)


# ============================================================
# Copia opcional a p22/data
# ============================================================

if COPY_TO_DATA_DIR:
    shutil.copy2(derived_dir / "Data_acc.npy", data_dir / "Data_acc.npy")
    shutil.copy2(derived_dir / "Data_adv.npy", data_dir / "Data_adv.npy")
    shutil.copy2(derived_dir / "Data_visc.npy", data_dir / "Data_visc.npy")

    shutil.copy2(derived_dir / "Data_acc_mks.npy", data_dir / "Data_acc_mks.npy")
    shutil.copy2(derived_dir / "Data_adv_mks.npy", data_dir / "Data_adv_mks.npy")
    shutil.copy2(derived_dir / "Data_visc_mks.npy", data_dir / "Data_visc_mks.npy")

    print("\nAlso copied to:")
    print(data_dir)


# ============================================================
# Chequeo rápido
# ============================================================

def print_stats(name, array):
    vals = array[:, :, 4:7]
    print("=" * 70)
    print(name)
    print("shape:", array.shape)
    print("NaN:", np.isnan(vals).sum())
    print("Inf:", np.isinf(vals).sum())
    print("min:", np.nanmin(vals))
    print("max:", np.nanmax(vals))
    print("mean:", np.nanmean(vals))
    print("std:", np.nanstd(vals))


print("\nQuick stats:")
print_stats("Data_acc_mks", Data_acc_mks)
print_stats("Data_adv_mks", Data_adv_mks)
print_stats("Data_visc_mks", Data_visc_mks)

print("\nDone.")