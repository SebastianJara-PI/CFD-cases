import numpy as np 

def train_test_split_numpy(X, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_train = int(train_ratio * len(X))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return X[train_idx], X[test_idx], train_idx, test_idx


def sample_batch(X, n_samples, rng):
    n_samples = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], idx


def normalize_dataset(
    X_raw,
    L,
    T,
    U,
    mu_x,
    mu_y,
    mu_z,
    mu_t,
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_t,
):

    X_space_a = X_raw[:, 0:3] / L
    X_time_a = X_raw[:, 3:4] / T
    X_vel_a = X_raw[:, 4:7] / U

    x_norm = (X_space_a[:, 0:1] - mu_x) / sigma_x
    y_norm = (X_space_a[:, 1:2] - mu_y) / sigma_y
    z_norm = (X_space_a[:, 2:3] - mu_z) / sigma_z
    t_norm = (X_time_a[:, 0:1] - mu_t) / sigma_t

    X_norm = np.concatenate(
        [
            x_norm,
            y_norm,
            z_norm,
            t_norm,
            X_vel_a,
        ],
        axis=1,
    )

    return X_norm



def prepare_pressure_training_data(
    Data_acc_mks,
    Data_adv_mks,
    Data_visc_mks,
    dic_adim_norm,
    L,
    T,
    U,
):
    """
    Construye un dataset para entrenar p_acc, p_adv, p_vis.

    Entrada:
        Data_acc_mks  : [Nt, Np, 7] = x,y,z,t, acc_x, acc_y, acc_z
        Data_adv_mks  : [Nt, Np, 7] = x,y,z,t, adv_x, adv_y, adv_z
        Data_visc_mks : [Nt, Np, 7] = x,y,z,t, vis_x, vis_y, vis_z

    Salida:
        X_pressure_tensor_norm : [N, 4]
        Y_grad_pressure        : [N, 9]

    donde Y_grad_pressure contiene:
        columnas 0:3 -> grad p_acc'
        columnas 3:6 -> grad p_adv'
        columnas 6:9 -> grad p_vis'
    """

    mu_x = float(np.asarray(dic_adim_norm["mu_x"]).squeeze())
    mu_y = float(np.asarray(dic_adim_norm["mu_y"]).squeeze())
    mu_z = float(np.asarray(dic_adim_norm["mu_z"]).squeeze())
    mu_t = float(np.asarray(dic_adim_norm["mu_t"]).squeeze())

    sigma_x = float(np.asarray(dic_adim_norm["sigma_x"]).squeeze())
    sigma_y = float(np.asarray(dic_adim_norm["sigma_y"]).squeeze())
    sigma_z = float(np.asarray(dic_adim_norm["sigma_z"]).squeeze())
    sigma_t = float(np.asarray(dic_adim_norm["sigma_t"]).squeeze())

    Nt, Np, _ = Data_acc_mks.shape

    Data_acc_flat = Data_acc_mks.reshape(-1, 7)
    Data_adv_flat = Data_adv_mks.reshape(-1, 7)
    Data_visc_flat = Data_visc_mks.reshape(-1, 7)

    coords = Data_acc_flat[:, 0:4]

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    z = coords[:, 2:3]
    t = coords[:, 3:4]

    x_norm = (x / L - mu_x) / sigma_x
    y_norm = (y / L - mu_y) / sigma_y
    z_norm = (z / L - mu_z) / sigma_z
    t_norm = (t / T - mu_t) / sigma_t

    X_norm = np.concatenate(
        [x_norm, y_norm, z_norm, t_norm],
        axis=1,
    )

    # ============================================================
    # Targets para gradientes adimensionales de presión
    # ============================================================

    scale = L / (U ** 2)

    grad_p_acc = -scale * Data_acc_flat[:, 4:7]
    grad_p_adv = -scale * Data_adv_flat[:, 4:7]
    grad_p_vis =  scale * Data_visc_flat[:, 4:7]

    Y_grad_pressure = np.concatenate(
        [
            grad_p_acc,
            grad_p_adv,
            grad_p_vis,
        ],
        axis=1,
    )

    return X_norm, Y_grad_pressure