import numpy as np
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
import matplotlib.pyplot as plt
from scipyDAE.radauDAE import RadauDAE
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
from config import m, g, l, phi0, t_span
import csv

#Рассматривается маятник, представляющий из себя сосредоточенную массу,
#подвешеную на невесомом стержне (подвесе)
#уравнения движения записываются через закон Ньютона с множителем Лагранжа для реализации реакции связи

#Уравнения движения формируются в виде
#M dX/dt = f(t,X)
#где M - матрица массово-инерционных характеристик (может быть вырожденной)
#f(t,X) - вектор правых частей

#ПАРАМЕТРЫ ЧИСЛЕННОГО ИНТЕГРАТОРА
rtol=1e-8; atol=rtol # параметры относительной и абсолютной точности
# dt_max = np.inf
bPrint=False # вывод дополнительных параметров
bDebug=False # вывод дополнительных параметров
method=RadauDAE

#ПАРАМЕТРЫ МАЯТНИКА
phi_dot0=0. #начальная угловая скорость

#Функция, формирующая правые и левые части уравнений
def generateSystem(phi_0=np.pi/2, phi_dot0=0., r=1., m=1, g=9.81):
    """ Возвращаемые значения:
          dae_fun: callable - вектор правых частей уравнений системы
          jac_dae: callable - якобиан правых частей уравнений системы
          mass: array_like - матрица массово-инерционных характеристик
          Xini: начальные данные интегрирования
    """

    def dae_fun(t, X):
        # X= (x,y,xdot=vx, ydot=vy, lbda)
        x = X[0]
        y = X[1]
        vx = X[2]
        vy = X[3]
        lam = X[4]
        return np.array([vx,
                         vy,
                         -x * lam / m,
                         -g - (y * lam) / m,
                         x ** 2 + y ** 2 - r ** 2])

    #матрица массово-инерционных характеристик, при том, что уравнения поделены на m
    mass = np.eye(5)
    mass[-1, -1] = 0
    #[1, 0, 0, 0, 0]
    #[0, 1, 0, 0, 0]
    #[0, 0, 1, 0, 0]
    #[0, 0, 0, 1, 0]
    #[0, 0, 0, 0, 0]
    var_index = np.array([0, 0, 0, 0, 3])

    def jac_dae(t, X):
        x = X[0]
        y = X[1]
        vx = X[2]
        vy = X[3]
        lam = X[4]
        return np.array([[0., 0., 1., 0., 0.],
                         [0., 0., 0., 1., 0.],
                         [-lam / m, 0., 0., 0., -x / m],
                         [0., -lam / m, 0., 0., -y / m],
                         [2 * x, 2 * y, 0., 0., 0.]])
    #формирование начальных данных
    #начальные данные должны быть согласованы с уравнениями связей!
    x0 =  r*np.sin(phi_0)
    y0 = -r*np.cos(phi_0)
    vx0 = r*phi_dot0*np.cos(phi_0)
    vy0 = r*phi_dot0*np.sin(phi_0)
    lam_0 = (m*r*phi_dot0**2 +  m*g*np.cos(phi_0))/r # equilibrium along the rod's axis
    Xini = np.array([x0,y0,vx0,vy0,lam_0])

    return dae_fun, jac_dae, mass, Xini, var_index

dae_fun, jac_dae, mass, Xini, var_index= generateSystem(phi0, phi_dot0, l, m, g)

#Решение системы ДАУ
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

# Сохранение данных в файл CSV
output_file = "log/D1_DAE_PointPendulum_Data.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Coordinate x (m)', 'Coordinate y (m)'])  # Заголовки столбцов
    for t, x, y in zip(sol.t, sol.y[0], sol.y[1]):
        writer.writerow([t, x, y])

print(f"Данные сохранены в файл: {output_file}")

#Получение массивов данных для переменных (включая алгебраическую переменную лямбда и реакцию подвеса T)
x = sol.y[0, :]
y = sol.y[1, :]
vx = sol.y[2, :]
vy = sol.y[3, :]
lam = sol.y[4, :]
T = lam * np.sqrt(x ** 2 + y ** 2)
#theta = computeAngle(x, y)  # np.arctan(x/y)

#Период колебаний
t_change = sol.t[:-1][np.diff(np.sign(vx)) < 0]
print('\tNumerical period={:.4e} s'.format(np.mean(np.diff(t_change))))

#Вывод графиков
fig, ax = plt.subplots(5,1,sharex=True, figsize=np.array([1.5,3])*5)
i=0
ax[i].plot(sol.t, x,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='x')
ax[i].plot(sol.t, y,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='y')
ax[i].set_ylim(-1.2*l, 1.2*l)
ax[i].legend(frameon=False)
ax[i].grid()
ax[i].set_ylabel('positions')

i+=1
ax[i].plot(sol.t, vx,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='vx')
ax[i].plot(sol.t, vy,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='vy')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel('velocities')

i+=1
ax[i].plot(sol.t, T,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='lam')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel('Lagrange multiplier\n(rod force)')

#метод Радо использует переменный шаг по времени
i+=1
ax[i].semilogy(sol.t[:-1], np.diff(sol.t), color='tab:blue', linestyle='-', marker='.', linewidth=1, label=r'$\Delta t$ (DAE)')
ax[i].grid()
ax[i].legend(frameon=False)
ax[i].set_ylabel(r'$\Delta t$ (s)')

i+=1
try:
    ax[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_real'],     color='tab:blue', linestyle='-', linewidth=2, label='cond(real) (DAE)')
    x[i].semilogy(sol.solver.info['cond']['t'],     sol.solver.info['cond']['LU_complex'],     color='tab:orange', linestyle='-', linewidth=1, label='cond(complexe) (DAE)')
except Exception as e:
    print(e)
    ax[i].legend(frameon=False)
    ax[i].grid()
    ax[i].set_ylabel('condition\nnumbers')

    ax[-1].set_xlabel('t (s)')

    t_eval = []
    for i in range(sol.t.size-1):
        t_eval.extend(  np.linspace(sol.t[i], sol.t[i+1], 20).tolist() )
    t_eval.append(sol.t[-1])
    t_eval = np.array(t_eval)

    dense_sol = sol.sol(t_eval)
    lam_dense = dense_sol[4,:]

    plt.figure()
    plt.plot(t_eval, lam_dense, label='dense output')
    plt.plot(sol.t, lam, label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[-1,:], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('lbda')
    plt.title('Dense output')

    ivar = 2
    plt.figure()
    plt.plot(t_eval, dense_sol[ivar,:], label='dense output')
    plt.plot(sol.t, sol.y[ivar,:], label='solution points', linestyle='', marker='o')
    plt.plot(sol.tsub, sol.ysub[ivar,:], label='sub solution points', linestyle='', marker='+', color='tab:red')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel(f'var {ivar}')
    plt.title('Dense output')

    plt.show()