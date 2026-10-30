import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
In this version, we consider the Solow Swan Model as an ODE form rather than in mapping form
This proves easier for dealing with variable population growth and technology growth

System Equations
We model initially the change in population as a linear increase over time
There the population will increase at an exponential rate over time
'''
def dLdt(L, n):
    return n * L

def cobb_douglas(K, L, A, alpha):
    return (K ** alpha) * (A * L)**(1 - alpha)

def dKdt(K, L, A, s, alpha, delta):
    return (s * cobb_douglas(K, L, A, alpha)) - (delta * K)

def dkdt(k, alpha, s, n, g, delta):
    return s * k**alpha - (n + g + delta) * k
'''
When k dot remains equal to 0 we reach steady state
So for some x where this is achieved, s * x**alpha = x * (n + g + delta)
At steady state, A * x**alpha = Y/L, and A (total factor productivity) grows at rate g with x constnat
Therefore, output per capita, Y/L grows at rate g
'''

def func_1(t, y, g, n, s, alpha, delta):
    (K, L, A, k) = (y[0], y[1], y[2], y[3])
    output_Y = (K**alpha) * (A*L) ** (1-alpha)
    dKdt = s * output_Y - delta * K
    dLdt = n * L
    dAdt = g * A
    dkdt = k**alpha - (n + g + delta) * k
    return (dKdt, dLdt, dAdt, dkdt)
'''
Initial Values
K0 - the initial value of total capital in Swindon. Found by getting the UK's total capital stock according to the Office of National statistics, and then taking Swindon's population fraction of that.
L0 - the current number of jobs in Swindon. Found from the Economic Growth Plan 2026-2031
A0 - the current total factor productivity. Found from Our World In Data, assuming the total factor productivity of the UK is not significantly different from Swindon
k0 - the initial total capital per effective worker. Found from K0/(A0 * L0)

System Parameters
g - the technologial growth rate. Found from Our World In Data and then taking changes across different time periods to find a good estimate
n - the population growth rate. Found from the Swindon Population data across year and finding changes across time periods for a good estimate
s - the share of output that is saved. World Bank reports savings rate at 0.18
alpha - the share of output from capital. This can be varied, but generally around 1/3
delta - the capital depreciation rate. Values generally between 0.05 and 0.1, so taken the median value of 0.08.
'''
K0 = 19600000000
L0 = 99600
A0 = 1
k0 = K0/(A0*L0)
g = 0.002
n = 0.005
s = 0.18
alpha = 0.35
delta = 0.08
y0 = (K0, L0, A0, k0)
t0 = 0
t_final = 20

solution = solve_ivp(func_1, [t0, t_final], y0, args = (g, n, s, alpha, delta))
output_per_worker = (solution.y[3])**alpha
k_star = (s/(n + g+ delta)) ** (1/(1-alpha))
'''
Testing to find a better plot for the steady state point
x_vals = np.where(solution.t < 2 * k_star)
y_vals = np.where(solution.y[3] < 2 * k)
'''
print(solution.y[3])
plt.plot(solution.y[3], output_per_worker)
plt.plot(solution.y[3], solution.y[3])
plt.xlabel("Capital per Effective Worker, $£$")
plt.ylabel("Output per Effective Worker, $£$")
plt.grid()
plt.title("An ODE Application of the Solow Swan Model")
plt.show()