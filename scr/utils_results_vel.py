import numpy as np
import matplotlib.pyplot as plt
import torch



def plot_velocity_loss_history(
    loss_history,
    best_iteration=None,
    stop_iteration=None,
):
    """
    Grafica:

    1. Pérdidas de velocidad sin ponderar.
    2. Pérdidas de velocidad ponderadas.
    3. Pesos adaptativos de cada componente.

    Parámetros
    ----------
    loss_history : dict
        Debe contener:
        - loss_data_u
        - loss_data_v
        - loss_data_w
        - lambda_data_u
        - lambda_data_v
        - lambda_data_w
    """

    # ============================
    # Loss sin ponderar
    # ============================

    loss_data_u = np.asarray(
        loss_history["loss_data_u"]
    )

    loss_data_v = np.asarray(
        loss_history["loss_data_v"]
    )

    loss_data_w = np.asarray(
        loss_history["loss_data_w"]
    )

    # ============================
    # Pesos adaptativos
    # ============================

    lambda_data_u = np.asarray(
        loss_history["lambda_data_u"]
    )

    lambda_data_v = np.asarray(
        loss_history["lambda_data_v"]
    )

    lambda_data_w = np.asarray(
        loss_history["lambda_data_w"]
    )

    # ============================
    # Loss ponderada
    # ============================

    loss_data_u_weighted = (
        lambda_data_u * loss_data_u
    )

    loss_data_v_weighted = (
        lambda_data_v * loss_data_v
    )

    loss_data_w_weighted = (
        lambda_data_w * loss_data_w
    )

    iterations = np.arange(len(loss_data_u))

    marker_specs = []

    if best_iteration is not None:
        marker_specs.append(
            (
                int(best_iteration),
                "--",
                "crimson",
                "Best checkpoint",
            )
        )

    if stop_iteration is not None:
        marker_specs.append(
            (
                int(stop_iteration),
                ":",
                "black",
                "Training stopped",
            )
        )

    # ============================
    # Figura con tres gráficos
    # ============================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 11),
        dpi=120,
        sharex=True,
    )

    # ============================
    # 1. Loss sin ponderar
    # ============================

    axes[0].semilogy(
        iterations,
        loss_data_u,
        label="Data U Loss",
        color="green",
        alpha=0.5,
    )

    axes[0].semilogy(
        iterations,
        loss_data_v,
        label="Data V Loss",
        color="blue",
        alpha=0.5,
    )

    axes[0].semilogy(
        iterations,
        loss_data_w,
        label="Data W Loss",
        color="orange",
        alpha=0.5,
    )

    axes[0].set_title(
        "Loss history",
        fontsize=18,
    )

    axes[0].set_ylabel(
        "Loss",
        fontsize=15,
    )

    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(
        axis="both",
        labelsize=13,
    )

    # ============================
    # 2. Loss ponderada
    # ============================

    axes[1].semilogy(
        iterations,
        loss_data_u_weighted,
        label="Weighted Data U Loss",
        color="green",
        alpha=0.5,
    )

    axes[1].semilogy(
        iterations,
        loss_data_v_weighted,
        label="Weighted Data V Loss",
        color="blue",
        alpha=0.5,
    )

    axes[1].semilogy(
        iterations,
        loss_data_w_weighted,
        label="Weighted Data W Loss",
        color="orange",
        alpha=0.5,
    )

    axes[1].set_title(
        "Weighted loss history",
        fontsize=18,
    )

    axes[1].set_ylabel(
        "Weighted loss",
        fontsize=15,
    )

    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(
        axis="both",
        labelsize=13,
    )

    # ============================
    # 3. Pesos adaptativos
    # ============================

    axes[2].semilogy(
        iterations,
        lambda_data_u,
        label=r"$\lambda_{data_u}$",
        color="green",
        alpha=0.7,
    )

    axes[2].semilogy(
        iterations,
        lambda_data_v,
        label=r"$\lambda_{data_v}$",
        color="blue",
        alpha=0.7,
    )

    axes[2].semilogy(
        iterations,
        lambda_data_w,
        label=r"$\lambda_{data_w}$",
        color="orange",
        alpha=0.7,
    )

    axes[2].set_title(
        "Adaptive loss weights",
        fontsize=18,
    )

    axes[2].set_xlabel(
        "Iteration",
        fontsize=15,
    )

    axes[2].set_ylabel(
        "Adaptive weights",
        fontsize=15,
    )

    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(
        axis="both",
        labelsize=13,
    )

    for axis in axes:
        for iteration, linestyle, color, label in marker_specs:
            axis.axvline(
                iteration,
                linestyle=linestyle,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=label,
            )
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig, axes



############################################################
def pred_pinns(data, net, factor_adim, L, T, U):
    # factor adim
    mu_x = factor_adim['mu_x']
    mu_y = factor_adim['mu_y']
    mu_z = factor_adim['mu_z']
    mu_t = factor_adim['mu_t']
    sigma_x = factor_adim['sigma_x']
    sigma_y = factor_adim['sigma_y']
    sigma_z = factor_adim['sigma_z']
    sigma_t = factor_adim['sigma_t']

    n_t, n_p, n_f = data.shape

    # Pinns predictions
    Dom_Data_flat = data.reshape(-1, n_f)[:, :4]

    Dom_Data_flat_tensor_norm = torch.cat([
        torch.tensor((Dom_Data_flat[:, 0:1] / L - mu_x) / sigma_x , dtype=torch.float64),
        torch.tensor((Dom_Data_flat[:, 1:2] / L - mu_y) / sigma_y, dtype=torch.float64),
        torch.tensor((Dom_Data_flat[:, 2:3] / L - mu_z) / sigma_z, dtype=torch.float64),
        torch.tensor((Dom_Data_flat[:, 3:4] / T - mu_t) / sigma_t, dtype=torch.float64)
    ], dim=1)

    # prediction Net
    x_dom = Dom_Data_flat_tensor_norm[:, 0:1]
    y_dom = Dom_Data_flat_tensor_norm[:, 1:2]
    z_dom = Dom_Data_flat_tensor_norm[:, 2:3]
    t_dom = Dom_Data_flat_tensor_norm[:, 3:4]

    Net_dom = net(x_dom, y_dom, z_dom, t_dom) 

    # Physics variables
    u_pred = Net_dom[:, 0:1]*U
    v_pred = Net_dom[:, 1:2]*U
    w_pred = Net_dom[:, 2:3]*U


    # coordenadas y tiempo desde Data original
    x_np = data[:, :, 0]
    y_np = data[:, :, 1]
    z_np = data[:, :, 2]
    t_np = data[:, :, 3]

    u_pred_np = u_pred.detach().cpu().numpy().reshape(n_t, n_p)
    v_pred_np = v_pred.detach().cpu().numpy().reshape(n_t, n_p)
    w_pred_np = w_pred.detach().cpu().numpy().reshape(n_t, n_p)

    # Velocity magnitude range
    vel_mag_np = np.sqrt(u_pred_np**2 + v_pred_np**2 + w_pred_np**2)
    vel_vmin = np.min(vel_mag_np)
    vel_vmax = np.max(vel_mag_np)

    print("Velocity magnitude range:", vel_vmin, vel_vmax)

    return x_np, y_np, z_np, t_np, u_pred_np, v_pred_np, w_pred_np, vel_mag_np, vel_vmin, vel_vmax




###########################################################
def plot_velocity_scatter_centerline(
    x_np,
    y_np,
    z_np,
    t_np,
    vel_mag_np,
    P1,
    P2,
    domain_points=None,
    centerline_points=None,
    time_step=7,
    azim=0,
    elev=90,
    vel_vmin=None,
    vel_vmax=None,
    alpha_dom=0.1,
    alpha_vel=1.0,
    alpha_centerline=1.0,
    point_size=1.0,
):


    # ============================================================
    # Conversión a NumPy
    # ============================================================

    def to_numpy(array):
        if hasattr(array, "detach"):
            return array.detach().cpu().numpy()

        return np.asarray(array)

    x_np = to_numpy(x_np)
    y_np = to_numpy(y_np)
    z_np = to_numpy(z_np)
    t_np = to_numpy(t_np)
    vel_mag_np = to_numpy(vel_mag_np)

    P1 = to_numpy(P1).reshape(-1)
    P2 = to_numpy(P2).reshape(-1)

    if domain_points is not None:
        domain_points = to_numpy(domain_points)

    if centerline_points is not None:
        centerline_points = to_numpy(centerline_points)

    # ============================================================
    # Validaciones simples
    # ============================================================

    n_time_steps = x_np.shape[0]

    if not 0 <= time_step < n_time_steps:
        raise ValueError(
            f"time_step debe estar entre 0 y {n_time_steps - 1}."
        )

    if P1.size < 3 or P2.size < 3:
        raise ValueError(
            "P1 y P2 deben contener al menos tres coordenadas."
        )

    # ============================================================
    # Datos del frame seleccionado
    # ============================================================

    x_t = x_np[time_step]
    y_t = y_np[time_step]
    z_t = z_np[time_step]

    vel_t = vel_mag_np[time_step]

    if t_np.ndim == 1:
        time_value = t_np[time_step]
    else:
        time_value = t_np[time_step, 0]

    # Escala de color común para todos los frames
    if vel_vmin is None:
        vel_vmin = np.nanmin(vel_mag_np)

    if vel_vmax is None:
        vel_vmax = np.nanmax(vel_mag_np)

    # ============================================================
    # Figura
    # ============================================================

    fig = plt.figure(
        figsize=(8, 6),
        dpi=120,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    sc = ax.scatter(
        x_t,
        y_t,
        z_t,
        c=vel_t,
        cmap="jet",
        vmin=vel_vmin,
        vmax=vel_vmax,
        s=point_size,
        alpha=alpha_vel,
    )

    # Puntos P1 y P2
    ax.scatter(
        P1[0],
        P1[1],
        P1[2],
        color="red",
        marker="x",
        s=120,
        linewidths=4,
        label=r"$\pi_1$",
    )

    ax.scatter(
        P2[0],
        P2[1],
        P2[2],
        color="black",
        marker="x",
        s=120,
        linewidths=4,
        label=r"$\pi_2$",
    )

    # Dominio completo en gris
    if domain_points is not None:
        ax.scatter(
            domain_points[:, 0],
            domain_points[:, 1],
            domain_points[:, 2],
            color="lightgray",
            alpha=alpha_dom,
            s=0.1,
        )

    # Centerline en azul
    if centerline_points is not None:
        ax.plot(
            centerline_points[:, 0],
            centerline_points[:, 1],
            centerline_points[:, 2],
            color="black",
            linewidth=1,
            alpha=alpha_centerline,
            label="Centerline",
        )

    ax.set_title(
        rf"$||\vec{{v}}||$ predicted | "
        rf"$t={time_value:.4f}$ s",
        fontsize=18,
    )

    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_zlabel("Z", fontsize=15)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
    )

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.75,
        pad=0.1,
    )

    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(
        r"$||\vec{v}||$",
        fontsize=20,
    )

    ax.legend(
        loc="upper right",
        fontsize=15,
    )

    ax.grid(False)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return fig, ax



#########################################
import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_comparison_centerline(
    Data,
    vel_mag_pred_np,
    P1,
    P2,
    domain_points=None,
    centerline_points=None,
    time_step=7,
    azim=0,
    elev=90,
    vel_vmin=None,
    vel_vmax=None,
    alpha_dom=0.1,
    alpha_vel=1.0,
    alpha_centerline=1.0,
    point_size=1.0,
):
    """
    Compara la magnitud de velocidad predicha por la PINN
    con la magnitud de velocidad real.

    Data debe tener columnas:

        [x, y, z, t, u_real, v_real, w_real]

    y shape:

        (Nt, Np, 7)
    """

    # ============================================================
    # Conversión a NumPy
    # ============================================================

    def to_numpy(array):
        if hasattr(array, "detach"):
            return array.detach().cpu().numpy()

        return np.asarray(array)


    Data = to_numpy(Data)
    vel_mag_pred_np = to_numpy(vel_mag_pred_np)

    P1 = to_numpy(P1).reshape(-1)
    P2 = to_numpy(P2).reshape(-1)

    if domain_points is not None:
        domain_points = to_numpy(domain_points)

    if centerline_points is not None:
        centerline_points = to_numpy(centerline_points)


    # ============================================================
    # Validaciones
    # ============================================================

    if Data.ndim != 3 or Data.shape[2] < 7:
        raise ValueError(
            "Data debe tener shape (Nt, Np, 7) y columnas "
            "[x, y, z, t, u, v, w]."
        )

    Nt, Np, _ = Data.shape

    if vel_mag_pred_np.shape != (Nt, Np):
        raise ValueError(
            "vel_mag_pred_np debe tener shape "
            f"{(Nt, Np)}, pero tiene {vel_mag_pred_np.shape}."
        )

    if not 0 <= time_step < Nt:
        raise ValueError(
            f"time_step debe estar entre 0 y {Nt - 1}."
        )


    # ============================================================
    # Coordenadas y tiempo
    # ============================================================

    x_np = Data[:, :, 0]
    y_np = Data[:, :, 1]
    z_np = Data[:, :, 2]
    t_np = Data[:, :, 3]


    # ============================================================
    # Velocidades reales
    # ============================================================

    u_real_np = Data[:, :, 4]
    v_real_np = Data[:, :, 5]
    w_real_np = Data[:, :, 6]

    vel_mag_real_np = np.sqrt(
        u_real_np**2
        + v_real_np**2
        + w_real_np**2
    )


    # ============================================================
    # Datos del frame
    # ============================================================

    x_t = x_np[time_step]
    y_t = y_np[time_step]
    z_t = z_np[time_step]

    vel_pred_t = vel_mag_pred_np[time_step]
    vel_real_t = vel_mag_real_np[time_step]

    time_value = t_np[time_step, 0]


    # ============================================================
    # Escala común de colores
    # ============================================================

    if vel_vmin is None:
        vel_vmin = np.nanmin(
            [
                np.nanmin(vel_mag_pred_np),
                np.nanmin(vel_mag_real_np),
            ]
        )

    if vel_vmax is None:
        vel_vmax = np.nanmax(
            [
                np.nanmax(vel_mag_pred_np),
                np.nanmax(vel_mag_real_np),
            ]
        )


    # ============================================================
    # Límites espaciales comunes
    # ============================================================

    xyz_all = Data[:, :, 0:3].reshape(-1, 3)

    xyz_min = np.nanmin(xyz_all, axis=0)
    xyz_max = np.nanmax(xyz_all, axis=0)

    spatial_range = np.maximum(
        xyz_max - xyz_min,
        1e-12,
    )


    # ============================================================
    # Figura
    # ============================================================

    fig = plt.figure(
        figsize=(15, 7),
        dpi=120,
        constrained_layout=True,
    )

    ax_pred = fig.add_subplot(
        1,
        2,
        1,
        projection="3d",
    )

    ax_real = fig.add_subplot(
        1,
        2,
        2,
        projection="3d",
    )

    axes = [ax_pred, ax_real]


    # ============================================================
    # Velocidad predicha
    # ============================================================

    sc_pred = ax_pred.scatter(
        x_t,
        y_t,
        z_t,
        c=vel_pred_t,
        cmap="jet",
        vmin=vel_vmin,
        vmax=vel_vmax,
        s=point_size,
        alpha=alpha_vel,
    )


    # ============================================================
    # Velocidad real
    # ============================================================

    ax_real.scatter(
        x_t,
        y_t,
        z_t,
        c=vel_real_t,
        cmap="jet",
        vmin=vel_vmin,
        vmax=vel_vmax,
        s=point_size,
        alpha=alpha_vel,
    )


    # ============================================================
    # Elementos comunes
    # ============================================================

    for ax in axes:

        # Dominio completo
        if domain_points is not None:
            ax.scatter(
                domain_points[:, 0],
                domain_points[:, 1],
                domain_points[:, 2],
                color="lightgray",
                alpha=alpha_dom,
                s=0.1,
            )

        # Centerline
        if centerline_points is not None:
            ax.plot(
                centerline_points[:, 0],
                centerline_points[:, 1],
                centerline_points[:, 2],
                color="black",
                linewidth=1,
                label="Centerline",
                alpha=alpha_centerline,
            )

        # Punto P1
        ax.scatter(
            P1[0],
            P1[1],
            P1[2],
            color="red",
            marker="x",
            s=120,
            linewidths=4,
            label=r"$\pi_1$",
        )

        # Punto P2
        ax.scatter(
            P2[0],
            P2[1],
            P2[2],
            color="black",
            marker="x",
            s=120,
            linewidths=4,
            label=r"$\pi_2$",
        )

        # Misma orientación
        ax.view_init(
            elev=elev,
            azim=azim,
        )

        # Mismos límites espaciales
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])

        ax.set_box_aspect(spatial_range)

        ax.grid(False)
        ax.set_axis_off()


    # ============================================================
    # Títulos
    # ============================================================

    ax_pred.set_title(
        rf"PINN predicted $||\vec{{v}}||$"
        rf" | $t={time_value:.4f}$ s",
        fontsize=18,
    )

    ax_real.set_title(
        rf"Real $||\vec{{v}}||$"
        rf" | $t={time_value:.4f}$ s",
        fontsize=18,
    )

    ax_pred.legend(
        loc="upper right",
        fontsize=12,
    )


    # ============================================================
    # Barra de colores compartida
    # ============================================================

    cbar = fig.colorbar(
        sc_pred,
        ax=axes,
        orientation="vertical",
        shrink=0.75,
        pad=0.03,
    )

    cbar.ax.tick_params(labelsize=14)

    cbar.set_label(
        r"$||\vec{v}||$ [m/s]",
        fontsize=18,
    )

    plt.show()

    return fig, axes



################################################################

def plot_velocity_relative_error(
    Data,
    u_pred_np,
    v_pred_np,
    w_pred_np,
    P1=None,
    P2=None,
    domain_points=None,
    centerline_points=None,
    time_step=7,
    azim=0,
    elev=90,
    min_real_velocity=1e-3,
    error_vmax=None,
    point_size=2.0,
    alpha_error=1.0,
    alpha_dom=0.1,
):
    """
    Grafica el error relativo vectorial de velocidad en cada punto:

        100 * ||v_pred - v_real||_2 / ||v_real||_2

    Data debe tener shape (Nt, Np, 7):

        [x, y, z, t, u_real, v_real, w_real]

    Los puntos donde ||v_real|| < min_real_velocity se excluyen
    para evitar errores relativos artificialmente grandes.
    """

    # ============================================================
    # Conversión a NumPy
    # ============================================================

    def to_numpy(array):
        if hasattr(array, "detach"):
            return array.detach().cpu().numpy()

        return np.asarray(array)

    Data = to_numpy(Data)

    u_pred_np = to_numpy(u_pred_np)
    v_pred_np = to_numpy(v_pred_np)
    w_pred_np = to_numpy(w_pred_np)

    if P1 is not None:
        P1 = to_numpy(P1).reshape(-1)

    if P2 is not None:
        P2 = to_numpy(P2).reshape(-1)

    if domain_points is not None:
        domain_points = to_numpy(domain_points)

    if centerline_points is not None:
        centerline_points = to_numpy(centerline_points)

    # ============================================================
    # Validaciones
    # ============================================================

    if Data.ndim != 3 or Data.shape[2] < 7:
        raise ValueError(
            "Data debe tener shape (Nt, Np, 7) con columnas "
            "[x, y, z, t, u, v, w]."
        )

    Nt, Np, _ = Data.shape

    expected_shape = (Nt, Np)

    for name, array in [
        ("u_pred_np", u_pred_np),
        ("v_pred_np", v_pred_np),
        ("w_pred_np", w_pred_np),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} debe tener shape {expected_shape}, "
                f"pero tiene {array.shape}."
            )

    if not 0 <= time_step < Nt:
        raise ValueError(
            f"time_step debe estar entre 0 y {Nt - 1}."
        )

    # ============================================================
    # Datos reales
    # ============================================================

    x_np = Data[:, :, 0]
    y_np = Data[:, :, 1]
    z_np = Data[:, :, 2]
    t_np = Data[:, :, 3]

    u_real_np = Data[:, :, 4]
    v_real_np = Data[:, :, 5]
    w_real_np = Data[:, :, 6]

    # ============================================================
    # Error vectorial absoluto
    # ============================================================

    velocity_error_norm = np.sqrt(
        (u_pred_np - u_real_np) ** 2
        + (v_pred_np - v_real_np) ** 2
        + (w_pred_np - w_real_np) ** 2
    )

    # Magnitud de la velocidad real
    velocity_real_norm = np.sqrt(
        u_real_np**2
        + v_real_np**2
        + w_real_np**2
    )

    # ============================================================
    # Error relativo [%]
    # ============================================================

    relative_error = np.full_like(
        velocity_real_norm,
        np.nan,
        dtype=float,
    )

    valid_points = velocity_real_norm >= min_real_velocity

    relative_error[valid_points] = (
        100.0
        * velocity_error_norm[valid_points]
        / velocity_real_norm[valid_points]
    )

    # ============================================================
    # Frame seleccionado
    # ============================================================

    x_t = x_np[time_step]
    y_t = y_np[time_step]
    z_t = z_np[time_step]

    error_t = relative_error[time_step]
    time_value = t_np[time_step, 0]

    valid_frame = np.isfinite(error_t)

    if not np.any(valid_frame):
        raise ValueError(
            "No hay puntos válidos en este frame. "
            "Prueba reduciendo min_real_velocity."
        )

    # El percentil evita que pocos outliers dominen la escala
    if error_vmax is None:
        error_vmax = np.nanpercentile(
            relative_error,
            99,
        )

    error_vmax = max(error_vmax, 1e-12)

    # ============================================================
    # Límites espaciales
    # ============================================================

    xyz_all = Data[:, :, 0:3].reshape(-1, 3)

    xyz_min = np.nanmin(xyz_all, axis=0)
    xyz_max = np.nanmax(xyz_all, axis=0)

    spatial_range = np.maximum(
        xyz_max - xyz_min,
        1e-12,
    )

    # ============================================================
    # Figura
    # ============================================================

    fig = plt.figure(
        figsize=(8, 6),
        dpi=120,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    sc = ax.scatter(
        x_t[valid_frame],
        y_t[valid_frame],
        z_t[valid_frame],
        c=error_t[valid_frame],
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        s=point_size,
        alpha=alpha_error,
    )

    # Dominio espacial
    if domain_points is not None:
        ax.scatter(
            domain_points[:, 0],
            domain_points[:, 1],
            domain_points[:, 2],
            color="lightgray",
            alpha=alpha_dom,
            s=0.1,
        )

    # Centerline
    if centerline_points is not None:
        ax.plot(
            centerline_points[:, 0],
            centerline_points[:, 1],
            centerline_points[:, 2],
            color="black",
            linewidth=1,
            label="Centerline",
        )

    # Puntos P1 y P2
    if P1 is not None:
        ax.scatter(
            P1[0],
            P1[1],
            P1[2],
            color="red",
            marker="x",
            s=120,
            linewidths=4,
            label=r"$\pi_1$",
        )

    if P2 is not None:
        ax.scatter(
            P2[0],
            P2[1],
            P2[2],
            color="black",
            marker="x",
            s=120,
            linewidths=4,
            label=r"$\pi_2$",
        )

    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])

    ax.set_box_aspect(spatial_range)

    ax.view_init(
        elev=elev,
        azim=azim,
    )

    ax.set_title(
        rf"Pointwise relative velocity error | "
        rf"$t={time_value:.4f}$ s",
        fontsize=18,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.75,
        pad=0.08,
    )

    cbar.ax.tick_params(labelsize=14)

    cbar.set_label(
        r"$E_{\mathrm{rel}}$ [\%]",
        fontsize=18,
    )

    if (
        centerline_points is not None
        or P1 is not None
        or P2 is not None
    ):
        ax.legend(
            loc="upper right",
            fontsize=12,
        )

    ax.grid(False)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return fig, ax, relative_error


#############################################
import numpy as np
import pandas as pd


def compute_velocity_metrics(
    Data,
    u_pred_np,
    v_pred_np,
    w_pred_np,
    min_real_velocity=1e-3,
):
    """
    Calcula métricas entre el campo de velocidad real y predicho.

    Parámetros
    ----------
    Data : array, shape (Nt, Np, 7)
        Columnas:
        [x, y, z, t, u_real, v_real, w_real]

    u_pred_np, v_pred_np, w_pred_np : array, shape (Nt, Np)
        Componentes de velocidad predichas por la PINN.

    min_real_velocity : float
        Velocidad mínima usada para calcular errores relativos
        y errores angulares. Evita divisiones por valores cercanos a cero.

    Retorna
    -------
    summary_df : pandas.DataFrame
        Métricas globales considerando todos los frames y puntos.

    frame_df : pandas.DataFrame
        Métricas calculadas separadamente en cada frame.
    """

    # ============================================================
    # Conversión a NumPy
    # ============================================================

    def to_numpy(array):
        if hasattr(array, "detach"):
            return array.detach().cpu().numpy()

        return np.asarray(array)

    Data = to_numpy(Data)

    u_pred_np = to_numpy(u_pred_np).squeeze()
    v_pred_np = to_numpy(v_pred_np).squeeze()
    w_pred_np = to_numpy(w_pred_np).squeeze()

    # ============================================================
    # Validaciones
    # ============================================================

    if Data.ndim != 3 or Data.shape[2] < 7:
        raise ValueError(
            "Data debe tener shape (Nt, Np, 7) con columnas "
            "[x, y, z, t, u, v, w]."
        )

    Nt, Np, _ = Data.shape
    expected_shape = (Nt, Np)

    for name, array in [
        ("u_pred_np", u_pred_np),
        ("v_pred_np", v_pred_np),
        ("w_pred_np", w_pred_np),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} debe tener shape {expected_shape}, "
                f"pero tiene {array.shape}."
            )

    # ============================================================
    # Campo real y campo predicho
    # ============================================================

    velocity_real = Data[:, :, 4:7]

    velocity_pred = np.stack(
        [
            u_pred_np,
            v_pred_np,
            w_pred_np,
        ],
        axis=2,
    )

    time_values = Data[:, 0, 3]

    # ============================================================
    # Función interna para calcular métricas
    # ============================================================

    def calculate_metrics(v_real, v_pred):

        v_real = np.asarray(v_real).reshape(-1, 3)
        v_pred = np.asarray(v_pred).reshape(-1, 3)

        # Eliminar puntos con NaN o infinito
        valid = (
            np.all(np.isfinite(v_real), axis=1)
            & np.all(np.isfinite(v_pred), axis=1)
        )

        v_real = v_real[valid]
        v_pred = v_pred[valid]

        if len(v_real) == 0:
            raise ValueError(
                "No existen puntos válidos para calcular las métricas."
            )

        error_vector = v_pred - v_real

        # Componentes
        mae_components = np.mean(
            np.abs(error_vector),
            axis=0,
        )

        rmse_components = np.sqrt(
            np.mean(error_vector**2, axis=0)
        )

        # ========================================================
        # Magnitudes
        # ========================================================

        speed_real = np.linalg.norm(
            v_real,
            axis=1,
        )

        speed_pred = np.linalg.norm(
            v_pred,
            axis=1,
        )

        speed_error = speed_pred - speed_real

        speed_mae = np.mean(
            np.abs(speed_error)
        )

        speed_rmse = np.sqrt(
            np.mean(speed_error**2)
        )

        speed_bias = np.mean(
            speed_error
        )

        # ========================================================
        # Error vectorial
        # ========================================================

        pointwise_vector_error = np.linalg.norm(
            error_vector,
            axis=1,
        )

        # Endpoint error medio
        epe_mean = np.mean(
            pointwise_vector_error
        )

        # RMSE vectorial
        vector_rmse = np.sqrt(
            np.mean(pointwise_vector_error**2)
        )

        # Error relativo L2 global
        denominator = np.linalg.norm(v_real)

        if denominator > 1e-12:
            relative_l2 = (
                100.0
                * np.linalg.norm(error_vector)
                / denominator
            )
        else:
            relative_l2 = np.nan

        # ========================================================
        # NRMSE de la magnitud
        # ========================================================

        speed_range = np.ptp(speed_real)

        if speed_range > 1e-12:
            speed_nrmse = (
                100.0
                * speed_rmse
                / speed_range
            )
        else:
            speed_nrmse = np.nan

        # ========================================================
        # R2 de la magnitud
        # ========================================================

        ss_res = np.sum(
            (speed_pred - speed_real) ** 2
        )

        ss_tot = np.sum(
            (speed_real - np.mean(speed_real)) ** 2
        )

        if ss_tot > 1e-12:
            speed_r2 = 1.0 - ss_res / ss_tot
        else:
            speed_r2 = np.nan

        # ========================================================
        # R2 de las componentes
        # ========================================================

        component_r2 = []

        for component in range(3):

            true_component = v_real[:, component]
            pred_component = v_pred[:, component]

            ss_res_component = np.sum(
                (pred_component - true_component) ** 2
            )

            ss_tot_component = np.sum(
                (
                    true_component
                    - np.mean(true_component)
                ) ** 2
            )

            if ss_tot_component > 1e-12:
                r2_component = (
                    1.0
                    - ss_res_component / ss_tot_component
                )
            else:
                r2_component = np.nan

            component_r2.append(r2_component)

        # ========================================================
        # Correlación de la magnitud
        # ========================================================

        if (
            np.std(speed_real) > 1e-12
            and np.std(speed_pred) > 1e-12
        ):
            speed_pearson = np.corrcoef(
                speed_real,
                speed_pred,
            )[0, 1]
        else:
            speed_pearson = np.nan

        # ========================================================
        # Error relativo puntual
        # ========================================================

        valid_relative = (
            speed_real >= min_real_velocity
        )

        if np.any(valid_relative):

            pointwise_relative_error = (
                100.0
                * pointwise_vector_error[valid_relative]
                / speed_real[valid_relative]
            )

            mean_relative_error = np.mean(
                pointwise_relative_error
            )

            median_relative_error = np.median(
                pointwise_relative_error
            )

            percentile_95_relative_error = np.percentile(
                pointwise_relative_error,
                95,
            )

        else:
            mean_relative_error = np.nan
            median_relative_error = np.nan
            percentile_95_relative_error = np.nan

        # ========================================================
        # Dirección de los vectores
        # ========================================================

        valid_direction = (
            (speed_real >= min_real_velocity)
            & (speed_pred >= min_real_velocity)
        )

        if np.any(valid_direction):

            dot_product = np.sum(
                v_real[valid_direction]
                * v_pred[valid_direction],
                axis=1,
            )

            cosine_similarity = (
                dot_product
                / (
                    speed_real[valid_direction]
                    * speed_pred[valid_direction]
                )
            )

            cosine_similarity = np.clip(
                cosine_similarity,
                -1.0,
                1.0,
            )

            angular_error = np.degrees(
                np.arccos(cosine_similarity)
            )

            mean_cosine_similarity = np.mean(
                cosine_similarity
            )

            mean_angular_error = np.mean(
                angular_error
            )

            median_angular_error = np.median(
                angular_error
            )

        else:
            mean_cosine_similarity = np.nan
            mean_angular_error = np.nan
            median_angular_error = np.nan

        # ========================================================
        # Peak de velocidad
        # ========================================================

        peak_speed_real = np.max(speed_real)
        peak_speed_pred = np.max(speed_pred)

        peak_speed_error = (
            peak_speed_pred - peak_speed_real
        )

        peak_speed_abs_error = np.abs(
            peak_speed_error
        )

        if peak_speed_real > 1e-12:
            peak_speed_relative_error = (
                100.0
                * peak_speed_abs_error
                / peak_speed_real
            )
        else:
            peak_speed_relative_error = np.nan

        return {
            "MAE u [m/s]": mae_components[0],
            "MAE v [m/s]": mae_components[1],
            "MAE w [m/s]": mae_components[2],

            "RMSE u [m/s]": rmse_components[0],
            "RMSE v [m/s]": rmse_components[1],
            "RMSE w [m/s]": rmse_components[2],

            "R2 u": component_r2[0],
            "R2 v": component_r2[1],
            "R2 w": component_r2[2],

            "Mean endpoint error [m/s]": epe_mean,
            "Vector RMSE [m/s]": vector_rmse,
            "Relative L2 error [%]": relative_l2,

            "Speed MAE [m/s]": speed_mae,
            "Speed RMSE [m/s]": speed_rmse,
            "Speed bias [m/s]": speed_bias,
            "Speed NRMSE range [%]": speed_nrmse,
            "Speed R2": speed_r2,
            "Speed Pearson r": speed_pearson,

            "Mean pointwise relative error [%]":
                mean_relative_error,

            "Median pointwise relative error [%]":
                median_relative_error,

            "P95 pointwise relative error [%]":
                percentile_95_relative_error,

            "Mean cosine similarity":
                mean_cosine_similarity,

            "Mean angular error [deg]":
                mean_angular_error,

            "Median angular error [deg]":
                median_angular_error,

            "Peak real speed [m/s]":
                peak_speed_real,

            "Peak predicted speed [m/s]":
                peak_speed_pred,

            "Peak speed absolute error [m/s]":
                peak_speed_abs_error,

            "Peak speed relative error [%]":
                peak_speed_relative_error,
        }

    # ============================================================
    # Métricas globales
    # ============================================================

    global_metrics = calculate_metrics(
        velocity_real,
        velocity_pred,
    )

    summary_df = pd.DataFrame(
        {
            "Metric": list(global_metrics.keys()),
            "Value": list(global_metrics.values()),
        }
    )

    # ============================================================
    # Métricas por frame
    # ============================================================

    frame_results = []

    for frame in range(Nt):

        frame_metrics = calculate_metrics(
            velocity_real[frame],
            velocity_pred[frame],
        )

        frame_results.append(
            {
                "Frame": frame,
                "Time [s]": time_values[frame],
                "Vector RMSE [m/s]":
                    frame_metrics["Vector RMSE [m/s]"],
                "Mean endpoint error [m/s]":
                    frame_metrics["Mean endpoint error [m/s]"],
                "Relative L2 error [%]":
                    frame_metrics["Relative L2 error [%]"],
                "Speed MAE [m/s]":
                    frame_metrics["Speed MAE [m/s]"],
                "Speed RMSE [m/s]":
                    frame_metrics["Speed RMSE [m/s]"],
                "Speed R2":
                    frame_metrics["Speed R2"],
                "Mean angular error [deg]":
                    frame_metrics["Mean angular error [deg]"],
                "Median pointwise relative error [%]":
                    frame_metrics[
                        "Median pointwise relative error [%]"
                    ],
            }
        )

    frame_df = pd.DataFrame(frame_results)

    return summary_df, frame_df




###########################################
def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def plot_density_panel(ax, x_est, y_ref, title,
                       xlabel="Estimated values",
                       ylabel="Reference values",
                       gridsize=120,
                       cmap="viridis",
                       ax_facecolor="#FFFFFF"):
    
    x_est = np.asarray(x_est).ravel()
    y_ref = np.asarray(y_ref).ravel()

    mask = np.isfinite(x_est) & np.isfinite(y_ref)
    x_est = x_est[mask]
    y_ref = y_ref[mask]

    xy_min = min(np.min(x_est), np.min(y_ref))
    xy_max = max(np.max(x_est), np.max(y_ref))

    # xy_min = np.min(y_ref)
    # xy_max = np.max(y_ref)

    ax.set_facecolor(ax_facecolor)

    hb = ax.hexbin(
        x_est, y_ref,
        gridsize=gridsize,
        bins='log',
        mincnt=1,
        cmap=cmap,
        linewidths=0
    )

    ax.plot([xy_min, xy_max], [xy_min, xy_max], 'r--', linewidth=1.5)

    r2 = r2_score_np(y_ref, x_est)

    ax.set_title(f"{title}\n$R^2 = {r2:.4f}$", fontsize=24)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)

    # ticks: min, intermedio negativo, 0, intermedio positivo, max
    neg_mid = xy_min / 2.0
    pos_mid = xy_max / 2.0

    ticks = [xy_min, neg_mid, 0.0, pos_mid, xy_max]

    # eliminar duplicados si xy_min o xy_max están muy cerca de 0
    ticks = np.array(ticks)
    ticks = np.unique(np.round(ticks, 10))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    return hb

def u_pred_pred_and_u_test(X_test, net, factor_adim, L, T, U):
    # factor adim
    mu_x = factor_adim['mu_x']
    mu_y = factor_adim['mu_y']
    mu_z = factor_adim['mu_z']
    mu_t = factor_adim['mu_t']
    sigma_x = factor_adim['sigma_x']
    sigma_y = factor_adim['sigma_y']
    sigma_z = factor_adim['sigma_z']
    sigma_t = factor_adim['sigma_t']

    X_test_tensor_norm = torch.cat([
    torch.tensor((X_test[:, 0:1] / L - mu_x)/sigma_x, dtype=torch.float64),
    torch.tensor((X_test[:, 1:2] / L - mu_y)/sigma_y, dtype=torch.float64),
    torch.tensor((X_test[:, 2:3] / L - mu_z)/sigma_z, dtype=torch.float64),
    torch.tensor((X_test[:, 3:4] / T - mu_t)/sigma_t, dtype=torch.float64)
        ], dim=1)

    x_test = X_test_tensor_norm[:, 0:1]
    y_test = X_test_tensor_norm[:, 1:2]
    z_test = X_test_tensor_norm[:, 2:3]
    t_test = X_test_tensor_norm[:, 3:4]

    Net_test = net(x_test, y_test, z_test, t_test)

    # predict test
    u_pred_test = Net_test[:, 0:1].detach().cpu().numpy().flatten() * U
    v_pred_test = Net_test[:, 1:2].detach().cpu().numpy().flatten() * U
    w_pred_test = Net_test[:, 2:3].detach().cpu().numpy().flatten() * U

    vel_pred_test = np.sqrt(u_pred_test**2 + v_pred_test**2 + w_pred_test**2)

    # real values test
    u_true_test = X_test[:, 4]
    v_true_test = X_test[:, 5]
    w_true_test = X_test[:, 6]

    vel_true_test = np.sqrt(u_true_test**2 + v_true_test**2 + w_true_test**2)

    return u_pred_test, v_pred_test, w_pred_test, vel_pred_test, u_true_test, v_true_test, w_true_test, vel_true_test




#########################################################
def compute_velocity_derivative_terms(
    model,
    Data,
    dic_adim_norm,
    L,
    T,
    U,
    Re,
    device,
    batch_size=20000,
):
    """
    Calcula los términos:
        acceleration = du/dt
        advective    = (u · grad)u
        viscous      = (1/Re) laplacian(u)

    sobre los mismos puntos espaciales y temporales de Data.

    Data debe tener shape:
        (n_time, n_points, n_features)

    y se asume que:
        Data[:, :, 0] = x
        Data[:, :, 1] = y
        Data[:, :, 2] = z
        Data[:, :, 3] = t

    Retorna:
        Data_acc      -> [x, y, z, t, acc_x, acc_y, acc_z]
        Data_adv      -> [x, y, z, t, adv_x, adv_y, adv_z]
        Data_visc     -> [x, y, z, t, visc_x, visc_y, visc_z]
    """

    model.eval()

    # ============================================================
    # Parámetros de normalización
    # ============================================================

    mu_x = float(dic_adim_norm["mu_x"])
    mu_y = float(dic_adim_norm["mu_y"])
    mu_z = float(dic_adim_norm["mu_z"])
    mu_t = float(dic_adim_norm["mu_t"])


    sigma_x = float(dic_adim_norm["sigma_x"])
    sigma_y = float(dic_adim_norm["sigma_y"])
    sigma_z = float(dic_adim_norm["sigma_z"])
    sigma_t = float(dic_adim_norm["sigma_t"])

    # ============================================================
    # Flatten de los datos
    # ============================================================

    n_time, n_points, n_features = Data.shape

    Data_flat = Data.reshape(-1, n_features)

    x = Data_flat[:, 0:1]
    y = Data_flat[:, 1:2]
    z = Data_flat[:, 2:3]
    t = Data_flat[:, 3:4]

    # Coordenadas originales que se guardarán en las estructuras finales
    coords_original = Data_flat[:, 0:4]

    # ============================================================
    # Normalización de inputs
    # ============================================================

    x_norm = (x / L - mu_x) / sigma_x
    y_norm = (y / L - mu_y) / sigma_y
    z_norm = (z / L - mu_z) / sigma_z
    t_norm = (t / T - mu_t) / sigma_t

    X_norm = np.concatenate(
        [x_norm, y_norm, z_norm, t_norm],
        axis=1,
    )

    # ============================================================
    # Arreglos de salida
    # ============================================================

    n_total = X_norm.shape[0]

    acc_all = np.zeros((n_total, 3))
    adv_all = np.zeros((n_total, 3))
    visc_all = np.zeros((n_total, 3))

    # ============================================================
    # Loop por batches
    # ============================================================

    for i in range(0, n_total, batch_size):
        print(f"Processing batch {i} to {min(i + batch_size, n_total)} of {n_total}")

        j = min(i + batch_size, n_total)

        X_batch_np = X_norm[i:j]

        X_batch = torch.tensor(
            X_batch_np,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        x_ = X_batch[:, 0:1]
        y_ = X_batch[:, 1:2]
        z_ = X_batch[:, 2:3]
        t_ = X_batch[:, 3:4]

        Net = model(x_, y_, z_, t_)

        u = Net[:, 0:1]
        v = Net[:, 1:2]
        w = Net[:, 2:3]

        # ========================================================
        # Primeras derivadas respecto a variables normalizadas
        # ========================================================

        u_x_hat = torch.autograd.grad(
            u, x_, grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_y_hat = torch.autograd.grad(
            u, y_, grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_z_hat = torch.autograd.grad(
            u, z_, grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_t_hat = torch.autograd.grad(
            u, t_, grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_x_hat = torch.autograd.grad(
            v, x_, grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_y_hat = torch.autograd.grad(
            v, y_, grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_z_hat = torch.autograd.grad(
            v, z_, grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_t_hat = torch.autograd.grad(
            v, t_, grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_x_hat = torch.autograd.grad(
            w, x_, grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_y_hat = torch.autograd.grad(
            w, y_, grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_z_hat = torch.autograd.grad(
            w, z_, grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_t_hat = torch.autograd.grad(
            w, t_, grad_outputs=torch.ones_like(w),
            create_graph=True,
            retain_graph=True,
        )[0]

        # ========================================================
        # Corrección por normalización de inputs
        # ========================================================
        # Como:
        #   x_hat = (x' - mu_x) / sigma_x
        # entonces:
        #   d/dx' = (1/sigma_x) d/dx_hat
        #
        # Aquí las derivadas quedan respecto a variables adimensionales
        # x' = x/L, t' = t/T.

        u_x = u_x_hat / sigma_x
        u_y = u_y_hat / sigma_y
        u_z = u_z_hat / sigma_z
        u_t = u_t_hat / sigma_t

        v_x = v_x_hat / sigma_x
        v_y = v_y_hat / sigma_y
        v_z = v_z_hat / sigma_z
        v_t = v_t_hat / sigma_t

        w_x = w_x_hat / sigma_x
        w_y = w_y_hat / sigma_y
        w_z = w_z_hat / sigma_z
        w_t = w_t_hat / sigma_t

        # ========================================================
        # Término de aceleración
        # ========================================================

        acc_x = u_t
        acc_y = v_t
        acc_z = w_t

        # ========================================================
        # Término advectivo
        # ========================================================

        adv_x = u * u_x + v * u_y + w * u_z
        adv_y = u * v_x + v * v_y + w * v_z
        adv_z = u * w_x + v * w_y + w * w_z

        # ========================================================
        # Segundas derivadas para término viscoso
        # ========================================================

        u_xx_hat = torch.autograd.grad(
            u_x_hat, x_,
            grad_outputs=torch.ones_like(u_x_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_yy_hat = torch.autograd.grad(
            u_y_hat, y_,
            grad_outputs=torch.ones_like(u_y_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_zz_hat = torch.autograd.grad(
            u_z_hat, z_,
            grad_outputs=torch.ones_like(u_z_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_xx_hat = torch.autograd.grad(
            v_x_hat, x_,
            grad_outputs=torch.ones_like(v_x_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_yy_hat = torch.autograd.grad(
            v_y_hat, y_,
            grad_outputs=torch.ones_like(v_y_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        v_zz_hat = torch.autograd.grad(
            v_z_hat, z_,
            grad_outputs=torch.ones_like(v_z_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_xx_hat = torch.autograd.grad(
            w_x_hat, x_,
            grad_outputs=torch.ones_like(w_x_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_yy_hat = torch.autograd.grad(
            w_y_hat, y_,
            grad_outputs=torch.ones_like(w_y_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        w_zz_hat = torch.autograd.grad(
            w_z_hat, z_,
            grad_outputs=torch.ones_like(w_z_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Corrección por normalización:
        # d2/dx'^2 = (1/sigma_x^2) d2/dx_hat^2

        u_xx = u_xx_hat / (sigma_x ** 2)
        u_yy = u_yy_hat / (sigma_y ** 2)
        u_zz = u_zz_hat / (sigma_z ** 2)

        v_xx = v_xx_hat / (sigma_x ** 2)
        v_yy = v_yy_hat / (sigma_y ** 2)
        v_zz = v_zz_hat / (sigma_z ** 2)

        w_xx = w_xx_hat / (sigma_x ** 2)
        w_yy = w_yy_hat / (sigma_y ** 2)
        w_zz = w_zz_hat / (sigma_z ** 2)

        lap_u = u_xx + u_yy + u_zz
        lap_v = v_xx + v_yy + v_zz
        lap_w = w_xx + w_yy + w_zz

        visc_x = (1.0 / Re) * lap_u
        visc_y = (1.0 / Re) * lap_v
        visc_z = (1.0 / Re) * lap_w

        # ========================================================
        # Guardar resultados
        # ========================================================

        acc_batch = torch.cat([acc_x, acc_y, acc_z], dim=1)
        adv_batch = torch.cat([adv_x, adv_y, adv_z], dim=1)
        visc_batch = torch.cat([visc_x, visc_y, visc_z], dim=1)

        acc_all[i:j, :] = acc_batch.detach().cpu().numpy()
        adv_all[i:j, :] = adv_batch.detach().cpu().numpy()
        visc_all[i:j, :] = visc_batch.detach().cpu().numpy()

    # ============================================================
    # Crear estructuras finales
    # ============================================================

    Data_acc_flat = np.concatenate(
        [coords_original, acc_all],
        axis=1,
    )

    Data_adv_flat = np.concatenate(
        [coords_original, adv_all],
        axis=1,
    )

    Data_visc_flat = np.concatenate(
        [coords_original, visc_all],
        axis=1,
    )

    Data_acc = Data_acc_flat.reshape(n_time, n_points, 7)
    Data_adv = Data_adv_flat.reshape(n_time, n_points, 7)
    Data_visc = Data_visc_flat.reshape(n_time, n_points, 7)

    return Data_acc, Data_adv, Data_visc