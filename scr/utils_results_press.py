import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_pressure_loss_history(
    loss_history_pres,
    figsize=(20, 5.5),
    dpi=120,
    alpha=0.6,
):
    """
    Muestra la evolución del entrenamiento de presión en tres paneles:

        1. Pérdidas sin ponderar.
        2. Pérdidas ponderadas.
        3. Pesos adaptativos.

    Parameters
    ----------
    loss_history_pres : dict
        Diccionario que debe contener:

            loss_acc
            loss_adv
            loss_vis
            lambda_acc
            lambda_adv
            lambda_vis

    figsize : tuple
        Tamaño de la figura.

    dpi : int
        Resolución de la figura.

    alpha : float
        Transparencia de las curvas.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura creada.

    axes : np.ndarray
        Arreglo con los tres ejes.
    """

    required_keys = [
        "loss_acc",
        "loss_adv",
        "loss_vis",
        "lambda_acc",
        "lambda_adv",
        "lambda_vis",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in loss_history_pres
    ]

    if missing_keys:
        raise KeyError(
            f"Faltan las siguientes claves en loss_history_pres: "
            f"{missing_keys}"
        )

    # ============================================================
    # Convertir historial a arrays
    # ============================================================

    loss_acc = np.asarray(
        loss_history_pres["loss_acc"],
        dtype=np.float64,
    )

    loss_adv = np.asarray(
        loss_history_pres["loss_adv"],
        dtype=np.float64,
    )

    loss_vis = np.asarray(
        loss_history_pres["loss_vis"],
        dtype=np.float64,
    )

    lambda_acc = np.asarray(
        loss_history_pres["lambda_acc"],
        dtype=np.float64,
    )

    lambda_adv = np.asarray(
        loss_history_pres["lambda_adv"],
        dtype=np.float64,
    )

    lambda_vis = np.asarray(
        loss_history_pres["lambda_vis"],
        dtype=np.float64,
    )

    lengths = {
        len(loss_acc),
        len(loss_adv),
        len(loss_vis),
        len(lambda_acc),
        len(lambda_adv),
        len(lambda_vis),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Las pérdidas y los pesos deben tener la misma longitud."
        )

    n_iterations = len(loss_acc)

    if n_iterations == 0:
        raise ValueError(
            "El historial de entrenamiento está vacío."
        )

    iterations = np.arange(n_iterations)

    # ============================================================
    # Pérdidas ponderadas
    # ============================================================

    loss_acc_weighted = lambda_acc * loss_acc
    loss_adv_weighted = lambda_adv * loss_adv
    loss_vis_weighted = lambda_vis * loss_vis

    # ============================================================
    # Crear figura
    # ============================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        dpi=dpi,
    )

    # ============================================================
    # Panel 1: pérdidas sin ponderar
    # ============================================================

    axes[0].semilogy(
        iterations,
        loss_acc,
        label=r"$\mathcal{L}_{acc}$",
        color="green",
        alpha=alpha,
    )

    axes[0].semilogy(
        iterations,
        loss_adv,
        label=r"$\mathcal{L}_{adv}$",
        color="blue",
        alpha=alpha,
    )

    axes[0].semilogy(
        iterations,
        loss_vis,
        label=r"$\mathcal{L}_{vis}$",
        color="orange",
        alpha=alpha,
    )

    axes[0].set_title(
        "Unweighted losses",
        fontsize=17,
    )

    axes[0].set_xlabel(
        "Iteration",
        fontsize=14,
    )

    axes[0].set_ylabel(
        "Loss",
        fontsize=14,
    )

    axes[0].legend(
        fontsize=10,
    )

    axes[0].grid(
        True,
        which="both",
        alpha=0.3,
    )

    axes[0].tick_params(
        axis="both",
        labelsize=12,
    )

    # ============================================================
    # Panel 2: pérdidas ponderadas
    # ============================================================

    axes[1].semilogy(
        iterations,
        loss_acc_weighted,
        label=r"$\lambda_{acc}\mathcal{L}_{acc}$",
        color="green",
        alpha=alpha,
    )

    axes[1].semilogy(
        iterations,
        loss_adv_weighted,
        label=r"$\lambda_{adv}\mathcal{L}_{adv}$",
        color="blue",
        alpha=alpha,
    )

    axes[1].semilogy(
        iterations,
        loss_vis_weighted,
        label=r"$\lambda_{vis}\mathcal{L}_{vis}$",
        color="orange",
        alpha=alpha,
    )

    axes[1].set_title(
        "Weighted losses",
        fontsize=17,
    )

    axes[1].set_xlabel(
        "Iteration",
        fontsize=14,
    )

    axes[1].set_ylabel(
        "Weighted loss",
        fontsize=14,
    )

    axes[1].legend(
        fontsize=10,
    )

    axes[1].grid(
        True,
        which="both",
        alpha=0.3,
    )

    axes[1].tick_params(
        axis="both",
        labelsize=12,
    )

    # ============================================================
    # Panel 3: pesos adaptativos
    # ============================================================

    axes[2].semilogy(
        iterations,
        lambda_acc,
        label=r"$\lambda_{acc}$",
        color="green",
        alpha=alpha,
    )

    axes[2].semilogy(
        iterations,
        lambda_adv,
        label=r"$\lambda_{adv}$",
        color="blue",
        alpha=alpha,
    )

    axes[2].semilogy(
        iterations,
        lambda_vis,
        label=r"$\lambda_{vis}$",
        color="orange",
        alpha=alpha,
    )

    axes[2].set_title(
        "Adaptive weights",
        fontsize=17,
    )

    axes[2].set_xlabel(
        "Iteration",
        fontsize=14,
    )

    axes[2].set_ylabel(
        r"$\lambda$",
        fontsize=14,
    )

    axes[2].legend(
        fontsize=10,
    )

    axes[2].grid(
        True,
        which="both",
        alpha=0.3,
    )

    axes[2].tick_params(
        axis="both",
        labelsize=12,
    )

    fig.tight_layout()

    return fig, axes



#######################################################

def predict_pressure_field(
    model,
    Data,
    dic_adim_norm,
    L,
    T,
    p0,
    device=None,
    batch_size=50000,
):
    """
    Predice las componentes de presión sobre todos los puntos de Data.

    La red debe representar:

        (x_norm, y_norm, z_norm, t_norm)
            -> (p_acc, p_adv, p_vis)

    donde las salidas de la red son presiones adimensionales.

    Parameters
    ----------
    model : torch.nn.Module
        Red neuronal entrenada para predecir:
        p_acc, p_adv y p_vis.

    Data : np.ndarray
        Arreglo con shape:

            (n_t, n_p, n_f)

        Las primeras cuatro columnas deben ser:

            [x, y, z, t]

    dic_adim_norm : dict
        Diccionario con:

            mu_x, mu_y, mu_z, mu_t
            sigma_x, sigma_y, sigma_z, sigma_t

    L : float
        Escala espacial característica, en metros.

    T : float
        Escala temporal característica, en segundos.

    p0 : float
        Escala de presión:

            p0 = rho * U**2

        en Pa.

    device : str or torch.device, optional
        Dispositivo de cálculo. Si es None, se usa el dispositivo
        donde se encuentra el modelo.

    batch_size : int
        Cantidad máxima de puntos procesados simultáneamente.

    Returns
    -------
    results : dict
        Diccionario con:

        - Data_pred_pres:
            [x, y, z, t, p_total] en mmHg.

        - Data_pred_components:
            [x, y, z, t, p_acc, p_adv, p_vis, p_total] en mmHg.

        - p_acc:
            Presión acelerativa con shape (n_t, n_p).

        - p_adv:
            Presión advectiva con shape (n_t, n_p).

        - p_vis:
            Presión viscosa con shape (n_t, n_p).

        - p_total:
            Presión total con shape (n_t, n_p).
    """

    # ============================================================
    # Validaciones
    # ============================================================

    Data = np.asarray(Data)

    if Data.ndim != 3:
        raise ValueError(
            "Data debe tener dimensiones (n_t, n_p, n_f). "
            f"Se recibió shape {Data.shape}."
        )

    if Data.shape[2] < 4:
        raise ValueError(
            "Data debe contener al menos las columnas [x, y, z, t]."
        )

    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")

    # ============================================================
    # Dimensiones originales
    # ============================================================

    n_t, n_p, _ = Data.shape
    n_total = n_t * n_p

    # ============================================================
    # Parámetros de normalización
    # ============================================================

    mu_x = float(np.asarray(dic_adim_norm["mu_x"]).squeeze())
    mu_y = float(np.asarray(dic_adim_norm["mu_y"]).squeeze())
    mu_z = float(np.asarray(dic_adim_norm["mu_z"]).squeeze())
    mu_t = float(np.asarray(dic_adim_norm["mu_t"]).squeeze())

    sigma_x = float(np.asarray(dic_adim_norm["sigma_x"]).squeeze())
    sigma_y = float(np.asarray(dic_adim_norm["sigma_y"]).squeeze())
    sigma_z = float(np.asarray(dic_adim_norm["sigma_z"]).squeeze())
    sigma_t = float(np.asarray(dic_adim_norm["sigma_t"]).squeeze())

    sigmas = np.array(
        [sigma_x, sigma_y, sigma_z, sigma_t],
        dtype=np.float64,
    )

    if np.any(sigmas <= 0):
        raise ValueError(
            "Todos los valores sigma deben ser estrictamente positivos."
        )

    # ============================================================
    # Dispositivo y tipo de datos del modelo
    # ============================================================

    try:
        model_parameter = next(model.parameters())
        model_device = model_parameter.device
        model_dtype = model_parameter.dtype
    except StopIteration:
        model_device = torch.device("cpu")
        model_dtype = torch.float64

    if device is None:
        device = model_device
    else:
        device = torch.device(device)
        model = model.to(device)

    # ============================================================
    # Coordenadas originales aplanadas
    # ============================================================

    X_flat = Data[:, :, :4].reshape(-1, 4)

    predictions = []

    model.eval()

    # ============================================================
    # Predicción por lotes
    # ============================================================

    with torch.inference_mode():

        for start in range(0, n_total, batch_size):

            end = min(start + batch_size, n_total)

            X_batch = X_flat[start:end]

            # Adimensionalización y normalización
            X_batch_norm = np.empty(
                X_batch.shape,
                dtype=np.float64,
            )

            X_batch_norm[:, 0] = (
                X_batch[:, 0] / L - mu_x
            ) / sigma_x

            X_batch_norm[:, 1] = (
                X_batch[:, 1] / L - mu_y
            ) / sigma_y

            X_batch_norm[:, 2] = (
                X_batch[:, 2] / L - mu_z
            ) / sigma_z

            X_batch_norm[:, 3] = (
                X_batch[:, 3] / T - mu_t
            ) / sigma_t

            X_batch_tensor = torch.as_tensor(
                X_batch_norm,
                dtype=model_dtype,
                device=device,
            )

            x_batch = X_batch_tensor[:, 0:1]
            y_batch = X_batch_tensor[:, 1:2]
            z_batch = X_batch_tensor[:, 2:3]
            t_batch = X_batch_tensor[:, 3:4]

            # La arquitectura recibe las cuatro variables separadas
            Net_batch = model(
                x_batch,
                y_batch,
                z_batch,
                t_batch,
            )

            if Net_batch.ndim != 2 or Net_batch.shape[1] < 3:
                raise ValueError(
                    "La red debe retornar un tensor con shape (N, 3), "
                    "correspondiente a [p_acc, p_adv, p_vis]. "
                    f"Se recibió {tuple(Net_batch.shape)}."
                )

            predictions.append(
                Net_batch[:, :3].detach().cpu().numpy()
            )

    # Unir todos los lotes
    Net_dom = np.concatenate(
        predictions,
        axis=0,
    )

    # ============================================================
    # Conversión de presión adimensional a mmHg
    # ============================================================

    pa_per_mmhg = 133.322

    pressure_scale_mmhg = p0 / pa_per_mmhg

    p_acc_flat = Net_dom[:, 0] * pressure_scale_mmhg
    p_adv_flat = Net_dom[:, 1] * pressure_scale_mmhg
    p_vis_flat = Net_dom[:, 2] * pressure_scale_mmhg

    # ============================================================
    # Recuperar la estructura temporal y espacial original
    # ============================================================

    p_acc = p_acc_flat.reshape(n_t, n_p)
    p_adv = p_adv_flat.reshape(n_t, n_p)
    p_vis = p_vis_flat.reshape(n_t, n_p)

    p_total = p_acc + p_adv + p_vis

    x_np = Data[:, :, 0]
    y_np = Data[:, :, 1]
    z_np = Data[:, :, 2]
    t_np = Data[:, :, 3]

    # [x, y, z, t, p_total]
    Data_pred_pres = np.stack(
        (
            x_np,
            y_np,
            z_np,
            t_np,
            p_total,
        ),
        axis=2,
    )

    # [x, y, z, t, p_acc, p_adv, p_vis, p_total]
    Data_pred_components = np.stack(
        (
            x_np,
            y_np,
            z_np,
            t_np,
            p_acc,
            p_adv,
            p_vis,
            p_total,
        ),
        axis=2,
    )

    return {
        "Data_pred_pres": Data_pred_pres,
        "Data_pred_components": Data_pred_components,
        "p_acc": p_acc,
        "p_adv": p_adv,
        "p_vis": p_vis,
        "p_total": p_total,
    }



#################################################################

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


def interactive_predicted_pressure(
    Data_pred_pres,
    idx_p1,
    idx_p2,
    P1_manual=None,
    P2_manual=None,
    Data_centerline=None,
    Data_seg=None,
    percentile_min=1,
    percentile_max=99,
    default_frame=7,
    default_azim=90,
    default_elev=20,
    default_point_size=5,
):
    """
    Visualización interactiva 3D del campo de presión predicho.

    Permite mostrar:

        1. Presión absoluta:
               p(x,t)

        2. Presión relativa respecto de P1:
               p(x,t) - p(P1,t)

    Parameters
    ----------
    Data_pred_pres : np.ndarray
        Arreglo con shape (n_t, n_p, 5):

            [x, y, z, t, p]

        La presión debe estar en mmHg.

    idx_p1 : int
        Índice del punto de referencia P1.

    idx_p2 : int
        Índice del punto P2 utilizado para calcular el drop:

            Δp = p(P1) - p(P2)

    P1_manual : array-like, optional
        Coordenadas [x, y, z] de P1 para mostrar el marcador.
        Si es None, se usan las coordenadas asociadas a idx_p1.

    P2_manual : array-like, optional
        Coordenadas [x, y, z] de P2 para mostrar el marcador.
        Si es None, se usan las coordenadas asociadas a idx_p2.

    Data_centerline : np.ndarray, optional
        Coordenadas de la centerline con shape (N, 3) o superior.

    Data_seg : np.ndarray, optional
        Coordenadas de la segmentación con shape (N, 3) o superior.

    percentile_min, percentile_max : float
        Percentiles utilizados para establecer los límites de color
        comunes a todos los frames.

    Returns
    -------
    interactive_widget
        Widget interactivo de ipywidgets.
    """

    # ============================================================
    # Validaciones
    # ============================================================

    Data_pred_pres = np.asarray(Data_pred_pres)

    if Data_pred_pres.ndim != 3:
        raise ValueError(
            "Data_pred_pres debe tener shape (n_t, n_p, n_f). "
            f"Shape recibido: {Data_pred_pres.shape}"
        )

    if Data_pred_pres.shape[2] < 5:
        raise ValueError(
            "Data_pred_pres debe contener como mínimo "
            "[x, y, z, t, presión]."
        )

    n_t, n_p, _ = Data_pred_pres.shape

    
    def normalize_index(idx, n_points, name):
        idx = int(idx)

        # Permitir índices negativos al estilo NumPy
        if idx < 0:
            idx = n_points + idx

        if not 0 <= idx < n_points:
            raise IndexError(
                f"{name} está fuera de rango. "
                f"Debe estar entre {-n_points} y {n_points - 1}."
            )

        return idx

    idx_p1 = normalize_index(
        idx=idx_p1,
        n_points=n_p,
        name="idx_p1",
    )

    idx_p2 = normalize_index(
        idx=idx_p2,
        n_points=n_p,
        name="idx_p2",
    )
    # ============================================================
    # Extraer datos
    # ============================================================

    xyz_pred = Data_pred_pres[:, :, 0:3]
    time_pred = Data_pred_pres[:, :, 3]
    pressure_absolute = Data_pred_pres[:, :, -1]

    # Presión relativa respecto de P1:
    # p(x,t) - p(P1,t)
    pressure_relative = (
        pressure_absolute
        - pressure_absolute[:, idx_p1][:, None]
    )

    # ============================================================
    # Coordenadas de P1 y P2
    # ============================================================

    if P1_manual is None:
        P1_plot = xyz_pred[0, idx_p1]
    else:
        P1_plot = np.asarray(P1_manual).reshape(3)

    if P2_manual is None:
        P2_plot = xyz_pred[0, idx_p2]
    else:
        P2_plot = np.asarray(P2_manual).reshape(3)

    # ============================================================
    # Escalas de color comunes a todos los frames
    # ============================================================

    def compute_color_limits(values):
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            raise ValueError(
                "No existen valores finitos para calcular "
                "los límites de color."
            )

        vmin = np.percentile(
            finite_values,
            percentile_min,
        )

        vmax = np.percentile(
            finite_values,
            percentile_max,
        )

        if np.isclose(vmin, vmax):
            delta = max(abs(vmin), 1.0) * 1e-12
            vmin -= delta
            vmax += delta

        return vmin, vmax

    vmin_absolute, vmax_absolute = compute_color_limits(
        pressure_absolute
    )

    vmin_relative, vmax_relative = compute_color_limits(
        pressure_relative
    )

    # ============================================================
    # Límites espaciales comunes
    # ============================================================

    xyz_flat = xyz_pred.reshape(-1, 3)

    xyz_min = np.nanmin(xyz_flat, axis=0)
    xyz_max = np.nanmax(xyz_flat, axis=0)

    spatial_range = np.maximum(
        xyz_max - xyz_min,
        1e-12,
    )

    # ============================================================
    # Función interna para graficar un frame
    # ============================================================

    def plot_frame(
        frame,
        pressure_mode,
        azim,
        elev,
        point_size,
        show_axis,
    ):
        xyz_frame = xyz_pred[frame]
        time_value = np.nanmedian(time_pred[frame])

        # Drop positivo en la dirección P1 -> P2
        delta_p = (
            pressure_absolute[frame, idx_p1]
            - pressure_absolute[frame, idx_p2]
        )

        if pressure_mode == "relative":

            pressure_frame = pressure_relative[frame]

            vmin = vmin_relative
            vmax = vmax_relative

            title_pressure = "Presión relativa predicha"

            colorbar_label = (
                r"$p(\mathbf{x},t)-p(P_1,t)$ [mmHg]"
            )

        elif pressure_mode == "absolute":

            pressure_frame = pressure_absolute[frame]

            vmin = vmin_absolute
            vmax = vmax_absolute

            title_pressure = "Presión predicha"

            colorbar_label = (
                r"$p(\mathbf{x},t)$ [mmHg]"
            )

        else:
            raise ValueError(
                "pressure_mode debe ser 'relative' o 'absolute'."
            )

        # ========================================================
        # Figura
        # ========================================================

        fig = plt.figure(
            figsize=(8, 6),
            dpi=120,
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
        )

        scatter = ax.scatter(
            xyz_frame[:, 0],
            xyz_frame[:, 1],
            xyz_frame[:, 2],
            c=pressure_frame,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=point_size,
            rasterized=True,
        )

        # ========================================================
        # Puntos P1 y P2
        # ========================================================

        ax.scatter(
            P1_plot[0],
            P1_plot[1],
            P1_plot[2],
            marker="x",
            s=120,
            linewidths=3,
            label="P1",
        )

        ax.scatter(
            P2_plot[0],
            P2_plot[1],
            P2_plot[2],
            marker="x",
            s=120,
            linewidths=3,
            label="P2",
        )

        # ========================================================
        # Centerline
        # ========================================================

        if Data_centerline is not None:

            centerline = np.asarray(Data_centerline)

            ax.plot(
                centerline[:, 0],
                centerline[:, 1],
                centerline[:, 2],
                color="black",
                linewidth=1,
                label="Centerline",
            )

        # ========================================================
        # Segmentación
        # ========================================================

        if Data_seg is not None:

            segmentation = np.asarray(Data_seg)

            ax.scatter(
                segmentation[:, 0],
                segmentation[:, 1],
                segmentation[:, 2],
                color="lightgray",
                alpha=0.8,
                s=0.1,
            )

        # ========================================================
        # Formato
        # ========================================================

        ax.set_xlim(
            xyz_min[0],
            xyz_max[0],
        )

        ax.set_ylim(
            xyz_min[1],
            xyz_max[1],
        )

        ax.set_zlim(
            xyz_min[2],
            xyz_max[2],
        )

        ax.set_box_aspect(spatial_range)

        ax.view_init(
            elev=elev,
            azim=azim,
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        ax.set_title(
            f"{title_pressure}\n"
            f"frame = {frame}, "
            f"t = {time_value:.4f} s, "
            rf"$\Delta p_{{P1\rightarrow P2}}$"
            f" = {delta_p:.3f} mmHg"
        )

        ax.legend()

        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.75,
            pad=0.08,
        )

        colorbar.set_label(
            colorbar_label,
            fontsize=12,
        )

        ax.grid(False)

        if not show_axis:
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()

    # ============================================================
    # Widget interactivo
    # ============================================================

    interactive_widget = widgets.interact(
        plot_frame,

        frame=widgets.IntSlider(
            value=min(default_frame, n_t - 1),
            min=0,
            max=n_t - 1,
            step=1,
            description="Frame:",
            continuous_update=False,
        ),

        pressure_mode=widgets.ToggleButtons(
            options=[
                ("Relativa a P1", "relative"),
                ("Sin referencia", "absolute"),
            ],
            value="relative",
            description="Presión:",
        ),

        azim=widgets.IntSlider(
            value=default_azim,
            min=-180,
            max=180,
            step=5,
            description="Azimut:",
            continuous_update=False,
        ),

        elev=widgets.IntSlider(
            value=default_elev,
            min=-90,
            max=90,
            step=5,
            description="Elevación:",
            continuous_update=False,
        ),

        point_size=widgets.FloatSlider(
            value=default_point_size,
            min=0.5,
            max=15,
            step=0.5,
            description="Tamaño:",
            continuous_update=False,
        ),

        show_axis=widgets.Checkbox(
            value=False,
            description="Mostrar ejes",
        ),
    )

    return interactive_widget


#####################################################
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_pressure_drop_decomposition(
    net_pres,
    Data,
    P1,
    P2,
    L,
    T,
    p0,
    mu_x,
    mu_y,
    mu_z,
    mu_t,
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_t,
    N_points=25,
    time_values=None,
    pressure_unit="mmHg",
    roll_shift=0,
    point_names=("P1", "P2"),
    title="Pressure prediction",
    figsize=(9, 10),
):
    """
    Evalúa una red de presión descompuesta en dos puntos espaciales
    durante un conjunto de instantes y genera dos paneles verticales.

    La red debe retornar al menos tres salidas:

        output[:, 0] = p_acc adimensional
        output[:, 1] = p_adv adimensional
        output[:, 2] = p_vis adimensional

    Parámetros
    ----------
    net_pres:
        Red neuronal de presión descompuesta.

    Data:
        Dataset con shape (Nt, Np, Nf).
        La columna Data[:, :, 3] debe contener el tiempo dimensional.

    P1, P2:
        Coordenadas espaciales con shape (3,), (1,3) o similar.

    L, T:
        Escalas características de longitud y tiempo.

    p0:
        Escala de presión, normalmente:

            p0 = rho * U**2

        Debe estar expresada en Pa.

    mu_x, ..., sigma_t:
        Parámetros usados para normalizar las entradas.

    N_points:
        Número de instantes a evaluar.

    time_values:
        Vector temporal opcional. Si es None, se construye entre
        el primer y último tiempo de Data.

    pressure_unit:
        "Pa" o "mmHg".

    roll_shift:
        Desplazamiento circular temporal aplicado a todas las curvas.

    point_names:
        Nombres utilizados en las etiquetas.

    Retorna
    -------
    results:
        Diccionario con presiones, componentes y drops.

    fig, axes:
        Figura y ejes de Matplotlib.
    """

    # ============================================================
    # Función auxiliar
    # ============================================================

    def to_numpy(array):
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()

        return np.asarray(array)

    # ============================================================
    # Validaciones y conversiones
    # ============================================================

    Data_np = to_numpy(Data)

    if Data_np.ndim != 3:
        raise ValueError(
            "Data debe tener shape (Nt, Np, Nf). "
            f"Shape recibido: {Data_np.shape}"
        )

    if Data_np.shape[2] < 4:
        raise ValueError(
            "Data debe contener al menos las columnas [x,y,z,t]."
        )

    P1_np = to_numpy(P1).reshape(-1)
    P2_np = to_numpy(P2).reshape(-1)

    if P1_np.size < 3 or P2_np.size < 3:
        raise ValueError(
            "P1 y P2 deben contener tres coordenadas espaciales."
        )

    P1_np = P1_np[:3]
    P2_np = P2_np[:3]

    L = float(np.asarray(L).squeeze())
    T = float(np.asarray(T).squeeze())
    p0 = float(np.asarray(p0).squeeze())

    mu_x = float(np.asarray(mu_x).squeeze())
    mu_y = float(np.asarray(mu_y).squeeze())
    mu_z = float(np.asarray(mu_z).squeeze())
    mu_t = float(np.asarray(mu_t).squeeze())

    sigma_x = float(np.asarray(sigma_x).squeeze())
    sigma_y = float(np.asarray(sigma_y).squeeze())
    sigma_z = float(np.asarray(sigma_z).squeeze())
    sigma_t = float(np.asarray(sigma_t).squeeze())

    if min(sigma_x, sigma_y, sigma_z, sigma_t) <= 0:
        raise ValueError(
            "Todos los valores sigma deben ser positivos."
        )

    # ============================================================
    # Vector temporal
    # ============================================================

    if time_values is None:

        t_initial = float(Data_np[0, 0, 3])
        t_final = float(Data_np[-1, 0, 3])

        t_p = np.linspace(
            t_initial,
            t_final,
            N_points,
        )

    else:

        t_p = to_numpy(time_values).reshape(-1)

        if t_p.size == 0:
            raise ValueError(
                "time_values no puede estar vacío."
            )

        N_points = t_p.size

    # ============================================================
    # Construir las entradas espacio-temporales
    # ============================================================

    def build_normalized_point_time(point):

        point_repeated = np.repeat(
            point[None, :],
            N_points,
            axis=0,
        )

        point_time = np.column_stack(
            [
                point_repeated,
                t_p,
            ]
        )

        point_time_norm = np.column_stack(
            [
                (
                    point_time[:, 0:1] / L
                    - mu_x
                ) / sigma_x,

                (
                    point_time[:, 1:2] / L
                    - mu_y
                ) / sigma_y,

                (
                    point_time[:, 2:3] / L
                    - mu_z
                ) / sigma_z,

                (
                    point_time[:, 3:4] / T
                    - mu_t
                ) / sigma_t,
            ]
        )

        return point_time_norm

    P1_xt_norm = build_normalized_point_time(
        P1_np
    )

    P2_xt_norm = build_normalized_point_time(
        P2_np
    )

    # ============================================================
    # Device y precisión de la red
    # ============================================================

    try:
        model_parameter = next(net_pres.parameters())

        model_device = model_parameter.device
        model_dtype = model_parameter.dtype

    except StopIteration:

        model_device = torch.device("cpu")
        model_dtype = torch.float64

    P1_xt_tensor = torch.as_tensor(
        P1_xt_norm,
        dtype=model_dtype,
        device=model_device,
    ).requires_grad_(True)

    P2_xt_tensor = torch.as_tensor(
        P2_xt_norm,
        dtype=model_dtype,
        device=model_device,
    ).requires_grad_(True)

    # ============================================================
    # Predicción
    # ============================================================

    net_pres.eval()

    # Se mantiene enable_grad por compatibilidad con arquitecturas
    # que calculan derivadas dentro del forward.
    with torch.enable_grad():

        Net_P1 = net_pres(
            P1_xt_tensor[:, 0:1],
            P1_xt_tensor[:, 1:2],
            P1_xt_tensor[:, 2:3],
            P1_xt_tensor[:, 3:4],
        )

        Net_P2 = net_pres(
            P2_xt_tensor[:, 0:1],
            P2_xt_tensor[:, 1:2],
            P2_xt_tensor[:, 2:3],
            P2_xt_tensor[:, 3:4],
        )

    if Net_P1.ndim != 2 or Net_P1.shape[1] < 3:
        raise ValueError(
            "La red debe retornar al menos tres columnas: "
            "[p_acc, p_adv, p_vis]. "
            f"Shape recibido: {tuple(Net_P1.shape)}"
        )

    if Net_P2.shape != Net_P1.shape:
        raise ValueError(
            "Las predicciones en P1 y P2 deben tener el mismo shape."
        )

    # ============================================================
    # Conversión a unidades físicas
    # ============================================================

    if pressure_unit.lower() == "mmhg":

        pressure_factor = p0 / 133.322
        pressure_label = "mmHg"

    elif pressure_unit.lower() == "pa":

        pressure_factor = p0
        pressure_label = "Pa"

    else:

        raise ValueError(
            "pressure_unit debe ser 'Pa' o 'mmHg'."
        )

    Net_P1_np = (
        Net_P1[:, 0:3]
        .detach()
        .cpu()
        .numpy()
        * pressure_factor
    )

    Net_P2_np = (
        Net_P2[:, 0:3]
        .detach()
        .cpu()
        .numpy()
        * pressure_factor
    )

    # Presiones por componente
    p1_acc = Net_P1_np[:, 0]
    p1_adv = Net_P1_np[:, 1]
    p1_vis = Net_P1_np[:, 2]

    p2_acc = Net_P2_np[:, 0]
    p2_adv = Net_P2_np[:, 1]
    p2_vis = Net_P2_np[:, 2]

    # Presión total en cada punto
    p1_total = (
        p1_acc
        + p1_adv
        + p1_vis
    )

    p2_total = (
        p2_acc
        + p2_adv
        + p2_vis
    )

    # ============================================================
    # Drops de presión
    # ============================================================

    delta_p_acc = p1_acc - p2_acc
    delta_p_adv = p1_adv - p2_adv
    delta_p_vis = p1_vis - p2_vis

    delta_p_total = (
        delta_p_acc
        + delta_p_adv
        + delta_p_vis
    )

    # ============================================================
    # Desplazamiento temporal opcional
    # ============================================================

    if roll_shift != 0:

        p1_total = np.roll(
            p1_total,
            roll_shift,
        )

        p2_total = np.roll(
            p2_total,
            roll_shift,
        )

        delta_p_acc = np.roll(
            delta_p_acc,
            roll_shift,
        )

        delta_p_adv = np.roll(
            delta_p_adv,
            roll_shift,
        )

        delta_p_vis = np.roll(
            delta_p_vis,
            roll_shift,
        )

        delta_p_total = np.roll(
            delta_p_total,
            roll_shift,
        )

    # ============================================================
    # Figura con dos paneles verticales
    # ============================================================

    name_P1, name_P2 = point_names

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Panel 1: presión en ambos puntos y drop total
    # ------------------------------------------------------------

    axes[0].plot(
        t_p,
        p1_total,
        linewidth=2.2,
        label=rf"$p({name_P1},t)$",
    )

    axes[0].plot(
        t_p,
        p2_total,
        linewidth=2.2,
        label=rf"$p({name_P2},t)$",
    )

    axes[0].plot(
        t_p,
        delta_p_total,
        linewidth=2.5,
        linestyle="--",
        label=(
            rf"$p({name_P1},t)"
            rf"-p({name_P2},t)$"
        ),
    )

    axes[0].axhline(
        0.0,
        linewidth=1,
        alpha=0.5,
    )

    axes[0].set_ylabel(
        f"Pressure [{pressure_label}]",
        fontsize=16,
    )

    axes[0].set_title(
        "Pressure at both points",
        fontsize=18,
    )

    axes[0].tick_params(
        axis="both",
        labelsize=13,
    )

    axes[0].grid(
        alpha=0.3,
    )

    axes[0].legend(
        fontsize=13,
    )

    # ------------------------------------------------------------
    # Panel 2: descomposición del drop
    # ------------------------------------------------------------

    axes[1].plot(
        t_p,
        delta_p_acc,
        linewidth=2,
        linestyle="--",
        label=r"$\Delta p_{\mathrm{acc}}$",
    )

    axes[1].plot(
        t_p,
        delta_p_adv,
        linewidth=2,
        linestyle="--",
        label=r"$\Delta p_{\mathrm{adv}}$",
    )

    axes[1].plot(
        t_p,
        delta_p_vis,
        linewidth=2,
        linestyle="--",
        label=r"$\Delta p_{\mathrm{vis}}$",
    )

    axes[1].plot(
        t_p,
        delta_p_total,
        linewidth=3,
        label=(
            r"$\Delta p_{\mathrm{acc}}"
            r"+\Delta p_{\mathrm{adv}}"
            r"+\Delta p_{\mathrm{vis}}$"
        ),
    )

    axes[1].axhline(
        0.0,
        linewidth=1,
        alpha=0.5,
    )

    axes[1].set_xlabel(
        "Time [s]",
        fontsize=16,
    )

    axes[1].set_ylabel(
        rf"$\Delta p$ [{pressure_label}]",
        fontsize=16,
    )

    axes[1].set_title(
        "Pressure-drop decomposition",
        fontsize=18,
    )

    axes[1].tick_params(
        axis="both",
        labelsize=13,
    )

    axes[1].grid(
        alpha=0.3,
    )

    axes[1].legend(
        fontsize=13,
        ncol=2,
    )

    fig.suptitle(
        title,
        fontsize=21,
    )

    plt.show()

    # ============================================================
    # Resultados
    # ============================================================

    results = {
        "time_s": t_p,

        "p1_acc": p1_acc,
        "p1_adv": p1_adv,
        "p1_vis": p1_vis,
        "p1_total": p1_total,

        "p2_acc": p2_acc,
        "p2_adv": p2_adv,
        "p2_vis": p2_vis,
        "p2_total": p2_total,

        "delta_p_acc": delta_p_acc,
        "delta_p_adv": delta_p_adv,
        "delta_p_vis": delta_p_vis,
        "delta_p_total": delta_p_total,

        "pressure_unit": pressure_label,
    }

    return results, fig, axes