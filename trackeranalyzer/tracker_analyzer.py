import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class TrackerAnalyzer:
    """
    Carga un CSV con columnas: t, x[, y[, z...]]
    y calcula velocidades y aceleraciones.
    Incluye estimación de valores faltantes usando Gaussian Process (MMSE).
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

    def plot_positions(self, interp_factor=10):
        """Grafica posición vs tiempo con puntos y línea interpolada."""
        t_min, t_max = self.t.min(), self.t.max()
        t_fine = np.linspace(t_min, t_max, len(self.t) * interp_factor)

        plt.figure(figsize=(12,4))
        for i, c in enumerate(self.pos_cols):
            plt.scatter(self.t, self.positions[:,i], s=20, label=f"{c} (estimado)")
            pos_i_fine = np.interp(t_fine, self.t, self.positions[:,i])
            plt.plot(t_fine, pos_i_fine, label=f"{c} (línea)")

        plt.xlabel('Tiempo (s)')
        plt.ylabel('Posición')
        plt.legend()
        plt.grid(True)
        plt.title('Posición vs Tiempo (con datos faltantes estimados)')
        plt.show()

    def plot_velocities(self):
        """Grafica velocidad vs tiempo."""
        plt.figure(figsize=(12,4))
        for i, c in enumerate(self.pos_cols):
            plt.plot(self.t, self.velocities[:,i], label=f"v_{c}(t)")
        if self.D > 1:
            plt.plot(self.t, self.speed, '--', label='|v|(t)')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Velocidad')
        plt.legend()
        plt.grid(True)
        plt.title('Velocidad vs Tiempo')
        plt.show()

    def plot_accelerations(self):
        """Grafica aceleración vs tiempo."""
        plt.figure(figsize=(12,4))
        for i, c in enumerate(self.pos_cols):
            plt.plot(self.t, self.accelerations[:,i], label=f"a_{c}(t)")
        if self.D > 1:
            plt.plot(self.t, self.accel, '--', label='|a|(t)')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Aceleración')
        plt.legend()
        plt.grid(True)
        plt.title('Aceleración vs Tiempo')
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

    def analyze_all(self, save_csv=None):
        """Grafica posición, velocidad y aceleración en secuencia.
        Si se pasa save_csv, guarda también el CSV resultante."""
        self.plot_positions()
        self.plot_velocities()
        self.plot_accelerations()
        self.to_dataframe()
        if save_csv:
            self.save_to_csv(save_csv)
