import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class TrackerAnalyzer:
    """
    Carga un CSV con columnas: t, x[, y[, z...]]
    Calcula velocidades y aceleraciones.
    Rellena valores faltantes (GP o interp) y normaliza posición inicial a cero.
    Permite exportar un DataFrame con t, posiciones, velocidades y aceleraciones.
    """
    def __init__(self, csv_file, skip_rows=1, fill_method='gp'):
        # Lectura y limpieza de columnas vacías
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')

        if 't' not in df.columns:
            raise ValueError("El archivo debe tener una columna 't' de tiempo.")
        self.t = df['t'].values

        # Detectar columnas de posición automáticamente
        self.pos_cols = [c for c in df.columns if c != 't']
        if not self.pos_cols:
            raise ValueError("No se detectaron columnas de posición.")
        self.raw_positions = df[self.pos_cols].values  # shape (N, D)
        self.D = self.raw_positions.shape[1]

        # Rellenar datos faltantes
        if fill_method == 'gp':
            self.positions = self._fill_gp(self.t, self.raw_positions)
        else:
            self.positions = self._fill_interp(self.t, self.raw_positions)

        # Normalizar la posición inicial a cero
        initial = self.positions[0].copy()
        self.positions = self.positions - initial

        # Calcular velocidades y aceleraciones
        self.velocities = np.gradient(self.positions, self.t, axis=0)
        self.accelerations = np.gradient(self.velocities, self.t, axis=0)
        if self.D > 1:
            self.speed = np.linalg.norm(self.velocities, axis=1)
            self.accel = np.linalg.norm(self.accelerations, axis=1)

    def _fill_interp(self, t, positions):
        """Interpolación lineal de valores faltantes (NaN)."""
        filled = np.empty_like(positions)
        for i in range(positions.shape[1]):
            xi = positions[:, i]
            mask = ~np.isnan(xi)
            filled[:, i] = np.interp(t, t[mask], xi[mask])
        return filled

    def _fill_gp(self, t, positions):
        """Usa GaussianProcessRegressor para estimar MMSE en cada dimensión."""
        filled = np.empty_like(positions)
        for i in range(positions.shape[1]):
            xi = positions[:, i]
            mask = ~np.isnan(xi)
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel,
                                          alpha=0.0,
                                          normalize_y=True)
            gp.fit(t[mask].reshape(-1,1), xi[mask])
            filled[:, i] = gp.predict(t.reshape(-1,1))
        return filled

    def analyze_all(self):
        """Grafica posición, velocidad y aceleración en subplots."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        markers = ['o', '^', 's']  # círculo, triángulo, cuadrado
        titles = ['Posición vs Tiempo', 'Velocidad vs Tiempo', 'Aceleración vs Tiempo']
        ylabels = ['Posición', 'Velocidad', 'Aceleración']
        font = {'fontname': 'DejaVu Sans', 'fontsize': 12}

        t_min, t_max = self.t.min(), self.t.max()
        t_fine = np.linspace(t_min, t_max, len(self.t) * 10)

        # Definir diferentes paletas de colores por gráfica
        cmap_pos = plt.cm.summer(np.linspace(0, 1, self.D))
        cmap_vel = plt.cm.plasma(np.linspace(0, 1, self.D))
        cmap_acc = plt.cm.Spectral(np.linspace(0, 1, self.D))

        # Posición
        for i, c in enumerate(self.pos_cols):
            pos_fine = np.interp(t_fine, self.t, self.positions[:, i])
            axes[0].scatter(self.t, self.positions[:, i], s=30, color=cmap_pos[i], label=f"${c}(t)$", marker=markers[0])
            axes[0].plot(t_fine, pos_fine, color=cmap_pos[i], linewidth=2.5)

        # Velocidad
        for i, c in enumerate(self.pos_cols):
            axes[1].scatter(self.t, self.velocities[:, i], s=30, color=cmap_vel[i], label=f"$v_{c}(t)$", marker=markers[1])
            axes[1].plot(self.t, self.velocities[:, i], color=cmap_vel[i], linewidth=2.5)

        # Aceleración
        for i, c in enumerate(self.pos_cols):
            axes[2].scatter(self.t, self.accelerations[:, i], s=30, color=cmap_acc[i], label=f"$a_{c}(t)$", marker=markers[2])
            axes[2].plot(self.t, self.accelerations[:, i], color=cmap_acc[i], linewidth=2.5)

        for ax, title, ylabel in zip(axes, titles, ylabels):
            ax.set_ylabel(ylabel, **font)
            ax.set_title(title, **font)
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel('Tiempo (s)', **font)
        plt.tight_layout()
        plt.show()

    def to_dataframe(self):
        """Devuelve un DataFrame con columnas: t, posiciones, velocidades y aceleraciones."""
        data = {'t': self.t}
        for i, c in enumerate(self.pos_cols):
            data[c] = self.positions[:, i]
            data[f'velocity_{c}'] = self.velocities[:, i]
            data[f'acceleration_{c}'] = self.accelerations[:, i]
        return pd.DataFrame(data)

    def save_to_csv(self, filename):
        """Guarda el DataFrame completo en un archivo CSV."""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)