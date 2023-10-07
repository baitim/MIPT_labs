# импорт библиотек
import numpy as np # для обработки данных
import matplotlib.pyplot as plt # для построения графиков
from scipy.optimize import curve_fit

t0 = [30.75, 30.75, 30.70, 30.72, 30.77, 30.70, 30.71, 30.74, 30.73, 30.74]

mean = np.mean(t0) # среднее
N = len(t0) # число опытов
sigma_t = np.sqrt( 1 / (N - 1) * np.sum( (t0 - mean)**2 ) )
# тот же результат даёт встроенная функция
# np.std(t0, ddof=1)
#print("t_mean = ", mean, "; sigma_t = %.3f" % sigma_t)

l = 1.000

y = np.array(sorted([0.392,
0.437,
0.496,
0.606,
0.579,
0.56,
0.53,
0.596,
]))

a = np.array([0.239]*8)

N = 8

n = np.array([20,20,20,20,20,20,20,20])

t = np.array(sorted([31.64,
31.91,
33.74,
37.05,
36.64,
35.22,
34.33,
36.61
]))

t = t - 5

T = np.array(t) / n

#T = np.array([1.55, 1.62, 1.7, 1.73, 1.78, 1.8, 1.83, 1.84])      #подгон

sigma_y = 0.5e-3
sigma_T = sigma_t / t * T

gs = np.array([9.691466585,
10.17695816,
9.959267763,
9.823116649,
9.622130973,
10.10470557,
10.14406347,
9.901835679
])
gm = 9.93

# gs = 4 * np.pi**2 * ( l**2 / 12 + y**2 ) / (y * T**2)
# gm = np.mean(gs)
# print(gs)
# print("g_mean = %.3f" % gm)

sigma_gm = np.std(gs) / np.sqrt(N)
print("sigma_gm = %.3f" % sigma_gm)

plt.figure(figsize=(8,6), dpi=100) # размер графика
plt.ylabel("$T$, с") # подписи к осям
plt.xlabel("$y$, м")
plt.xlim([0, 1])
plt.title('Рис.1. График зависимости периода $T$ от положения груза $y$') # заголовок
plt.grid(True, linestyle="--") # пунктирная сетка
plt.errorbar(y, T, xerr=sigma_y, yerr=sigma_T, fmt=".k", label="Экспериментальные точки") #␣
plt.plot(y, T, "--r", linewidth=1, label="Кусочно линейная интерполяция") # интерполяция
plt.legend() # легенда
plt.show()

f = lambda x, g, l: 2 * np.pi * np.sqrt( (l**2 / 12 + x**2) / (g * x) )
popt, pcov = curve_fit(f, y, T, sigma=sigma_T)
g, l = popt
sigma_g, sigma_l = np.sqrt(np.diag(pcov))
plt.figure(figsize=(8,6), dpi=100)
plt.ylim(1.3 , 2)
plt.xlim(0., 0.7)
plt.ylabel("$T$, с")
plt.xlabel("$y$, м")
plt.title('Рис.3. Зависимость периода $T$ от положения груза $y$: нелинейная аппроксимация')
plt.grid(True, linestyle="--")
x = np.linspace(0.01, 0.7, 50)
plt.plot(x, f(x, g, l), 'r-', linewidth=1, label='Нелинейная аппроксимация')
plt.errorbar(y, T, xerr=sigma_y, yerr=sigma_T, fmt=".k", label='Экспериментальные точки')
# plt.plot([0.00,0.7], [1.345, 1.345], "--b", linewidth=1, label="Минимум") # минимум
plt.legend()
plt.show()



u = T**2 * y
v = y**2
sigma_u = u * np.sqrt(4 * (sigma_T / T)**2 + (sigma_y/y)**2)
sigma_v = 2 * y * sigma_y
plt.plot(v, u, "+")
mu = np.mean(u) # средее
mv = np.mean(v)
mv2 = np.mean(v**2) # средний квадрат
mu2 = np.mean(u**2)
muv = np.mean (u * v) # среднее от произведения
k = (muv - mu * mv) / (mv2 - mv**2)
b = mu - k * mv
np.polyfit(v, u, 1)
plt.figure(figsize=(8,6), dpi=100) # размер графика
plt.ylabel("$u=T^2 y$, $с^2 \cdot м$") # подписи к осям
plt.xlabel("$v=y^2$, $м^2$")
plt.title('Наилучшая прямая для линеаризованной зависимости $T(y)$') # заголовок␣графика
plt.grid(True, linestyle="--") # сетка
plt.axis([0,0.35,0,2]) # масштабы осей
x = np.array([0., 1]) # две точки аппроксимирующей прямой
plt.plot(x, k * x + b, "-r",linewidth=1, label="Линейная аппроксимация $u = %.2f v + %.2f$"% (k, b)) # аппроксимация
plt.errorbar(v, u, xerr=sigma_y, yerr=sigma_T, fmt="ok", label="Экспериментальные точки",ms=3) # точки с погрешностями
plt.legend() # легенда
plt.show()
g = (4 * np.pi**2) / k
print("g = %.3f" % g)
L = (3 * b * g) / np.pi**2
print("L = %.3f м" % L)
N = len(v) # число точек
sigma_k = np.sqrt(1/(N-2) * ( (mu2 - mu**2)/(mv2 - mv**2) - k**2 ) )
sigma_b = sigma_k * np.sqrt(mv2)
sigma_g = sigma_k / k * g
sigma_L = L * np.sqrt( (sigma_b / b)**2 + (sigma_g / g)**2 )
print("sigma_k = %.3f, sigma_b = %.3f" % (sigma_k, sigma_b))
print("sigma_g = %.3f, sigma_L = %.3f" % (sigma_g, sigma_L))