import csv
import numpy as np
from numpy.ma.core import arcsin
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
import matplotlib.pyplot as plt
from scipyDAE.radauDAE import RadauDAE
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
from config import m, g, phi0, l, t_span

# Рассматривается плоский маятник, представляющий из себя твёрдое тело
# уравнения движения записываются через закон Ньютона с множителем Лагранжа для реализации реакции связи

# Уравнения движения формируются в виде
# M dX/dt = f(t,X)
# где M - матрица массово-инерционных характеристик (может быть вырожденной)
# f(t,X) - вектор правых частей

# ПАРАМЕТРЫ ЧИСЛЕННОГО ИНТЕГРАТОРА
rtol = 1e-8
atol = rtol  # параметры относительной и абсолютной точности
# dt_max = np.inf
bPrint = False  # вывод дополнительных параметров
bDebug = False  # вывод дополнительных параметров
method = RadauDAE

# ПАРАМЕТРЫ МАЯТНИКА
phi_dot0 = 0.  # начальная угловая скорость
half_l = 1.5  # расстояние от точки подвеса до центра масс
J = m*half_l**2. / 3.  # момент инерции относительно оси, проходящей через центр масс
# маятник считается однородным стержнем, подвешеным за один из концов

# Функция, формирующая правые и левые части уравнений


def generateSystem(phi_0=np.pi/2, phi_dot0=0., l=1., m=1, J=1./12., g=9.81):
    """ Возвращаемые значения:
          dae_fun: callable - вектор правых частей уравнений системы
          jac_dae: callable - якобиан правых частей уравнений системы
          mass: array_like - матрица массово-инерционных характеристик
          Xini: начальные данные интегрирования
    """

    def dae_fun(t, X):
        # X = [x, y, phi, vx, vy, omega, lam1, lam2]
        x = X[0]
        y = X[1]
        phi = X[2]
        vx = X[3]
        vy = X[4]
        omega = X[5]
        lam1 = X[6]
        lam2 = X[7]

        # Правые части уравнений
        dx_dt = vx
        dy_dt = vy
        dphi_dt = omega
        dvx_dt = -lam1 / m
        dvy_dt = -g - lam2 / m
        domega_dt = (l * np.cos(phi) * lam1 + l * np.sin(phi) * lam2) / J

        # Алгебраические уравнения для множителей Лагранжа
        # x^2 + y^2 - l^2 = 0
        # phi - arcsin(x / l) = 0
        constraint1 = x**2 + y**2 - l**2
        constraint2 = phi - np.arcsin(x / l)

        return np.array([
            dx_dt,
            dy_dt,
            dphi_dt,
            dvx_dt,
            dvy_dt,
            domega_dt,
            constraint1,
            constraint2
        ])

    # матрица массово-инерционных характеристик, при том, что уравнения поделены на m и J
    mass = np.eye(8)
    mass[-1, -1] = 0
    mass[-2, -2] = 0
    # [1, 0, 0, 0, 0, 0, 0, 0]
    # [0, 1, 0, 0, 0, 0, 0, 0]
    # [0, 0, 1, 0, 0, 0, 0, 0]
    # [0, 0, 0, 1, 0, 0, 0, 0]
    # [0, 0, 0, 0, 1, 0, 0, 0]
    # [0, 0, 0, 0, 0, 1, 0, 0]
    # [0, 0, 0, 0, 0, 0, 0, 0]
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var_index = np.array([0, 0, 0, 0, 0, 0, 3, 3])  # указывается индекс алгебраической переменной

    def jac_dae(t, X):
        x = X[0]
        y = X[1]
        phi = X[2]
        vx = X[3]
        vy = X[4]
        omega = X[5]
        lam1 = X[6]
        lam2 = X[7]

        # Векторные производные для переменных
        d_dx_dt = [0, 0, 0, 1, 0, 0, 0, 0]
        d_dy_dt = [0, 0, 0, 0, 1, 0, 0, 0]
        d_dphi_dt = [0, 0, 0, 0, 0, 1, 0, 0]
        d_dvx_dt = [0, 0, 0, 0, 0, 0, -1 / m, 0]
        d_dvy_dt = [0, 0, 0, 0, 0, 0, 0, -1 / m]
        d_domega_dt = [
            0, 0, -(l * np.sin(phi) * lam1 + l * np.cos(phi) * lam2) / J, 0, 0, 0,
            l * np.cos(phi) / J, l * np.sin(phi) / J
        ]

        # Частные производные ограничений
        d_constraint1 = [2 * x, 2 * y, 0, 0, 0, 0, 0, 0]
        d_constraint2 = [-1 / (l * np.sqrt(1 - (x / l)**2)), 0, 1, 0, 0, 0, 0, 0]

        # Формируем матрицу Якоби
        jacobian = np.array([
            d_dx_dt,
            d_dy_dt,
            d_dphi_dt,
            d_dvx_dt,
            d_dvy_dt,
            d_domega_dt,
            d_constraint1,
            d_constraint2
        ])

        return jacobian

    # формирование начальных данных
    # начальные данные должны быть согласованы с уравнениями связей!
    x0 = l*np.sin(phi_0)
    y0 = -l*np.cos(phi_0)
    phi0 = phi_0
    vx0 = l*phi_dot0*np.cos(phi_0)
    vy0 = l*phi_dot0*np.sin(phi_0)
    dphi0 = phi_dot0
    # для нахождения начальных значений множителей Лагранжа lam1 и lam2
    # нужно решить относительно них систему ДУ при условии равновесия (ускорения = 0)
    lam10 = -(m*g)/y0
    lam20 = -(x0*m*g*l*(1-x0**2./l**2.)**0.5)/y0
    Xini = np.array([x0, y0, phi0, vx0, vy0, dphi0, lam10, lam20])

    return dae_fun, jac_dae, mass, Xini, var_index


dae_fun, jac_dae, mass, Xini, var_index = generateSystem(phi0, phi_dot0, half_l, m, J, g)

# Решение системы ДАУ
print(f'Solving the index {3} formulation')
sol = solve_ivp(fun=dae_fun, t_span=t_span, y0=Xini, max_step=t_span[1] / 10,
                rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=None,
                method=method, vectorized=False, first_step=1e-3, dense_output=True,
                mass_matrix=mass, bPrint=bPrint, return_substeps=True,
                max_newton_ite=10, min_factor=0.2, max_factor=10,
                var_index=var_index,
                newton_tol=1e-2,  # TODO: depending on each variable's index ?
                scale_residuals=True,
                scale_newton_norm=False,
                scale_error=True,
                max_bad_ite=1,
                bDebug=bDebug)
print("DAE of index {} {} in {} time steps, {} fev, {} jev, {} LUdec".format(
    3, 'solved' * sol.success + (1 - sol.success) * 'failed',
    sol.t.size, sol.nfev, sol.njev, sol.nlu))

# Получение массивов данных для переменных (включая алгебраическую переменную лямбда и реакцию подвеса T)
x = sol.y[0, :]
y = sol.y[1, :]
phi = sol.y[2, :]
vx = sol.y[3, :]
vy = sol.y[4, :]
dphi = sol.y[5, :]
lam1 = sol.y[6, :]
lam2 = sol.y[7, :]
R = lam1 * np.sqrt(x ** 2 + y ** 2)
# theta = computeAngle(x, y)  # np.arctan(x/y)

# Период колебаний
t_change = sol.t[:-1][np.diff(np.sign(vx)) < 0]
print('\tNumerical period={:.4e} s'.format(np.mean(np.diff(t_change))))

# Вывод графиков
fig, ax = plt.subplots(5, 1, sharex=True, figsize=np.array([1.5, 3])*5)
i = 0
ax[i].plot(sol.t, x,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='x')
ax[i].plot(sol.t, y,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='y')
ax[i].plot(sol.t, phi,     color='tab:red', linestyle='-', linewidth=2, marker='.', label='phi')
ax[i].set_ylim(-1.2*half_l, 1.2*half_l)
ax[i].legend(frameon=False)
ax[i].grid()
ax[i].set_ylabel('positions')

i += 1
ax[i].plot(sol.t, vx,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='vx')
ax[i].plot(sol.t, vy,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='vy')
ax[i].plot(sol.t, dphi,     color='tab:red', linestyle='-', linewidth=2, marker='.', label='dphi')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel('velocities')

i += 1
ax[i].plot(sol.t, R,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='lam1')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel('Lagrange multiplier\n(rod force)')

# метод Радо использует переменный шаг по времени
i += 1
ax[i].semilogy(sol.t[:-1], np.diff(sol.t), color='tab:blue', linestyle='-',
               marker='.', linewidth=1, label=r'$\Delta t$ (DAE)')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel(r'$\Delta t$ (s)')

i += 1
try:
    ax[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_real'],
                   color='tab:blue', linestyle='-', linewidth=2, label='cond(real) (DAE)')
    x[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_complex'],
                  color='tab:orange', linestyle='-', linewidth=1, label='cond(complexe) (DAE)')
except Exception as e:
    print(e)
    ax[i].legend(frameon=False)
    ax[i].grid()
    ax[i].set_ylabel('condition\nnumbers')

    ax[-1].set_xlabel('t (s)')

    t_eval = []
    for i in range(sol.t.size-1):
        t_eval.extend(np.linspace(sol.t[i], sol.t[i+1], 20).tolist())
    t_eval.append(sol.t[-1])
    t_eval = np.array(t_eval)

    dense_sol = sol.sol(t_eval)
    lam1_dense = dense_sol[6, :]

    plt.figure()
    plt.plot(t_eval, lam1_dense, label='dense output')
    plt.plot(sol.t, lam1, label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[-1, :], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('lbda')
    plt.title('Dense output')

    ivar = 2
    plt.figure()
    plt.plot(t_eval, dense_sol[ivar, :], label='dense output')
    plt.plot(sol.t, sol.y[ivar, :], label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[ivar, :], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel(f'var {ivar}')
    plt.title('Dense output')

    plt.show()


# Сохранение данных в файл CSV

output_file = "log/D2_DAE_BodyPendulum_Data.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Coordinate x (m)', 'Coordinate y (m)'])  # Заголовки столбцов

    # Проходим по всем точкам в решении
    for t, x, y, phi in zip(sol.t, sol.y[0], sol.y[1], sol.y[2]):
        # Сдвигаем координаты в зависимости от угла отклонения маятника
        x_shifted = x + half_l * np.sin(phi)
        y_shifted = y - half_l * np.cos(phi)

        # Записываем в файл сдвигнутые координаты
        writer.writerow([t, x_shifted, y_shifted])

print(f"Данные сохранены в файл: {output_file}")
