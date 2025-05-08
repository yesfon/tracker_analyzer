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
    Permite ajustar la curva de posición a y(t)=At^2+Bt+C, estimar g=2A y R^2.
    Permite exportar un DataFrame con t, posiciones, velocidades y aceleraciones.
    """
    def __init__(self, csv_file, skip_rows=1, fill_method='gp'):
        df = pd.read_csv(csv_file, skiprows=skip_rows)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        if 't' not in df.columns:
            raise ValueError("El archivo debe tener una columna 't' de tiempo.")
        self.t = df['t'].values
        self.pos_cols = [c for c in df.columns if c != 't']
        if not self.pos_cols:
            raise ValueError("No se detectaron columnas de posición.")
        self.raw_positions = df[self.pos_cols].values
        self.D = self.raw_positions.shape[1]
        if fill_method == 'gp':
            self.positions = self._fill_gp(self.t, self.raw_positions)
        else:
            self.positions = self._fill_interp(self.t, self.raw_positions)
        initial = self.positions[0].copy()
        self.positions -= initial
        self.velocities = np.gradient(self.positions, self.t, axis=0)
        self.accelerations = np.gradient(self.velocities, self.t, axis=0)
        if self.D > 1:
            self.speed = np.linalg.norm(self.velocities, axis=1)
            self.accel = np.linalg.norm(self.accelerations, axis=1)

    def _fill_interp(self, t, positions):
        filled = np.empty_like(positions)
        for i in range(positions.shape[1]):
            xi = positions[:, i]
            mask = ~np.isnan(xi)
            filled[:, i] = np.interp(t, t[mask], xi[mask])
        return filled

    def _fill_gp(self, t, positions):
        filled = np.empty_like(positions)
        for i in range(positions.shape[1]):
            xi = positions[:, i]
            mask = ~np.isnan(xi)
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
            gp.fit(t[mask].reshape(-1,1), xi[mask])
            filled[:, i] = gp.predict(t.reshape(-1,1))
        return filled

    def analyze_all(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        markers = ['o', '^', 's']
        titles = ['Posición vs Tiempo', 'Velocidad vs Tiempo', 'Aceleración vs Tiempo']
        ylabels = ['Posición', 'Velocidad', 'Aceleración']
        font = {'fontname': 'DejaVu Sans', 'fontsize': 12}
        t_min, t_max = self.t.min(), self.t.max()
        t_fine = np.linspace(t_min, t_max, len(self.t) * 10)
        cmap_pos = plt.cm.viridis(np.linspace(0, 1, self.D))
        cmap_vel = plt.cm.plasma(np.linspace(0, 1, self.D))
        cmap_acc = plt.cm.inferno(np.linspace(0, 1, self.D))

        # POSICIÓN: ajuste y anotaciones en esquina superior derecha
        for i, c in enumerate(self.pos_cols):
            y = self.positions[:, i]
            coeffs = np.polyfit(self.t, y, 2)
            A, B, C = coeffs
            y_fit = np.polyval(coeffs, self.t)
            ss_res = np.sum((y - y_fit)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot!=0 else np.nan
            g_est = 2 * A
            axes[0].scatter(self.t, y, s=30, color=cmap_pos[i], label=f"${c}(t)$", marker=markers[0])
            axes[0].plot(t_fine, np.polyval(coeffs, t_fine), color=cmap_pos[i], linewidth=2.5)
            axes[0].text(0.98, 0.2 - i*0.08,
                         f"$y(t)={C:.2f} + {B:.2f}t + 1/2({A:.2f})t^2$",
                         transform=axes[0].transAxes,
                         ha='right', va='top',
                         fontname='DejaVu Sans', fontsize=12)
            axes[0].text(0.98, 0.1 - i*0.08,
                         f"$g={g_est:.2f} m/s², R²={r2:.2f}$",
                         transform=axes[0].transAxes,
                         ha='right', va='top',
                         fontname='DejaVu Sans', fontsize=12)

        # VELOCIDAD
        for i, c in enumerate(self.pos_cols):
            axes[1].scatter(self.t, self.velocities[:, i], s=30, color=cmap_vel[i], label=f"$v_{c}(t)$", marker=markers[1])
            axes[1].plot(self.t, self.velocities[:, i], color=cmap_vel[i], linewidth=2.5)

        # ACELERACIÓN
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
        data = {'t': self.t}
        for i, c in enumerate(self.pos_cols):
            data[c] = self.positions[:, i]
            data[f'velocity_{c}'] = self.velocities[:, i]
            data[f'acceleration_{c}'] = self.accelerations[:, i]
        return pd.DataFrame(data)

    def save_to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)