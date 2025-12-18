import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import csv
from config import g, l, m, t_span, phi0

# Определяем систему уравнений
def pendulum(t, y):
    phi, v = y  # y[0] = φ, y[1] = v
    dphi_dt = v
    dv_dt = - (3 / 2) * (g / l) * np.sin(phi)
    return [dphi_dt, dv_dt]

# Начальные условия
v0 = 0.0          # начальная угловая скорость
y0 = [phi0, v0]

# Временной диапазон для решения
t_eval = np.linspace(t_span[0], t_span[1], 500)  # точки для вывода результата

# Решение системы
solution = solve_ivp(pendulum, t_span, y0, t_eval=t_eval, method='RK45')

# Сохранение данных в файл CSV
output_file = "log/D2_GC_BodyPendulum_Data.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Angle φ (rad)'])  # Заголовки столбцов
    for t, phi in zip(solution.t, solution.y[0]):
        writer.writerow([t, phi])

print(f"Данные сохранены в файл: {output_file}")

# Построение графиков
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='φ(t) (угол)')
plt.plot(solution.t, solution.y[1], label='v(t) (угловая скорость)')
plt.legend()
plt.xlabel('Время (с)')
plt.ylabel('Значение')
plt.title('Динамика математического маятника')
plt.grid()
plt.show()
