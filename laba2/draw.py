import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

# Параметры маятника
l = 3.0  # длина маятника, м

paths = [
    'log/D1_AN_PointPendulum_Data.csv',
    'log/D1_DAE_PointPendulum_Data.csv',
    'log/D2_DAE_BodyPendulum_Data.csv',
    'log/D1_GC_PointPendulum_Data.csv',
    'log/D2_GC_BodyPendulum_Data.csv'
]

def read_data(path: str):
    """Чтение данных из CSV файла"""
    time, x_coordinate, y_coordinate = [], [], []

    with open(path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаем заголовок

        for row in reader:
            row = [float(i) for i in row]
            t = row[0]
            if len(row) == 3:
                x, y = row[1], row[2]
            else:
                phi = row[1]
                x = l * math.sin(phi)
                y = -l * math.cos(phi)

            time.append(t)
            x_coordinate.append(x)
            y_coordinate.append(y)

    return time, x_coordinate, y_coordinate

def interpolate(x, y):
    """Создание интерполяционной функции"""
    return interp1d(x, y, kind='linear', fill_value='extrapolate')

def animate_multiple_pendulums(functions_dict, t_range=(0, 10), dt=0.05):
    time_data = np.arange(t_range[0], t_range[1], dt)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-l - 0.5, l + 0.5)
    ax.set_ylim(-l - 0.5, l + 0.5)
    ax.set_aspect('equal')
    ax.grid()

    lines = {name: ax.plot([], [], 'o-', lw=2, label=name)[0] for name in functions_dict}
    ax.legend()

    def init():
        for line in lines.values():
            line.set_data([], [])
        return lines.values()

    def update(frame):
        t = time_data[frame]
        for name, func in functions_dict.items():
            x, y = func(t)
            lines[name].set_data([0, x], [0, y])
        return lines.values()

    ani = animation.FuncAnimation(
        fig, update, frames=len(time_data), init_func=init, blit=True, interval=20
    )

    plt.title("Анимация нескольких маятников")
    plt.show()

def plot_2d_pendulums(functions_dict, t_range=(0, 10), dt=0.05):
    time_data = np.arange(t_range[0], t_range[1], dt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.set_xlabel("Время (с)")
    ax1.set_ylabel("X координата (м)")
    ax1.set_title("Зависимость X от времени")

    ax2.set_xlabel("Время (с)")
    ax2.set_ylabel("Y координата (м)")
    ax2.set_title("Зависимость Y от времени")

    for name, func in functions_dict.items():
        x_data, y_data = [], []
        for t in time_data:
            x, y = func(t)
            x_data.append(x)
            y_data.append(y)

        # Строим график для координаты x на первом подграфике
        ax1.plot(time_data, x_data, label=name)
        # Строим график для координаты y на втором подграфике
        ax2.plot(time_data, y_data, label=name)

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Интерполяционные функции
    point_interpolated_laws = {}
    body_interpolated_laws = {}

    # Чтение данных из файлов и интерполяция
    for path in paths:
        time, x, y = read_data(path)
        fx_interpolated = interpolate(time, x)
        fy_interpolated = interpolate(time, y)

        # Создание функций с возвратом координат
        if 'Point' in path:
            point_interpolated_laws[path] = lambda t, fx=fx_interpolated, fy=fy_interpolated: (fx(t), fy(t))
        if 'Body' in path:
            body_interpolated_laws[path] = lambda t, fx=fx_interpolated, fy=fy_interpolated: (fx(t), fy(t))

    # Запуск визуализации
    animate_multiple_pendulums(point_interpolated_laws, t_range=(0, 10), dt=0.05)
    plot_2d_pendulums(point_interpolated_laws, t_range=(0, 10), dt=0.05)

    animate_multiple_pendulums(body_interpolated_laws, t_range=(0, 10), dt=0.05)
    plot_2d_pendulums(body_interpolated_laws, t_range=(0, 10), dt=0.05)

if __name__ == "__main__":
    main()
