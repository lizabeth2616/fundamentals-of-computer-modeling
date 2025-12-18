import numpy as np
import matplotlib.pyplot as plt
import csv
from config import g, l, phi0, t_span

# Временной диапазон для решения
t_eval = np.linspace(t_span[0], t_span[1], 500)  # точки для вывода результата

# Аналитическое решение
omega = np.sqrt(g / l)
phi_analytic = phi0 * np.cos(omega * t_eval)

# Сохранение данных в файл CSV
output_file = "log/D1_AN_PointPendulum_Data.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Angle φ (rad)'])  # Заголовки столбцов
    for t, phi in zip(t_eval, phi_analytic):
        writer.writerow([t, phi])

print(f"Данные сохранены в файл: {output_file}")

# Построение графиков
plt.figure(figsize=(10, 5))
plt.plot(t_eval, phi_analytic, label='φ(t) (угол)')
plt.legend()
plt.xlabel('Время (с)')
plt.ylabel('Значение')
plt.title('Динамика математического маятника (аналитическое решение)')
plt.grid()
plt.show()
