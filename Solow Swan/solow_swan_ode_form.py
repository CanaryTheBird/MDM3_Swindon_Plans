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
A0 - the current total factor productivity. Found using K0, L0, alpha and the Cobb Douglas production function
k0 - the initial total capital per effective worker. Found from K0/(A0 * L0)

System Parameters
g - the technologial growth rate. Found from Our World In Data and then taking changes across different time periods to find a good estimate
n - the population growth rate. Found from the Swindon Population data across year and finding changes across time periods for a good estimate
s - the share of output that is saved. World Bank reports savings rate at 0.18
alpha - the share of output from capital. This can be varied, but generally around 1/3
delta - the capital depreciation rate. Values generally between 0.05 and 0.1, so taken the median value of 0.08.
'''
K0 = 12503623082
L0 = 155927
Y0 = 12700000000
g = 0.002
n = 0.005
s = 0.18
alpha = 0.35
delta = 0.08
A0 = (Y0 / (K0**alpha * L0**(1 - alpha))) ** (1 / (1 - alpha))
k0 = K0/(A0*L0)
y0 = (K0, L0, A0, k0)
t_range = np.linspace(0, 100, 500)

solution = solve_ivp(func_1, [0, 100], y0, args = (g, n, s, alpha, delta), t_eval=t_range)
output_per_worker = (solution.y[3])**alpha
k_star = (s/(n + g+ delta)) ** (1/(1-alpha))
k_grid = np.linspace(0, 2 * max(k_star, k0), 400)
savings = s * k_grid ** alpha
break_even = (n + g + delta) * k_grid
print(k0)
fig, ax = plt.subplots(1)
'''Plots for the savings and break even line'''
#ax.plot(k_grid, break_even, label = "Break Even Investment")
#ax.plot(k_grid, savings, label = "Savings Line")
#ax.vlines(k_star, 0, 1.5, color="black", linestyles= "dashed", label = f"Steady State at $k^* = ${k_star}")
#ax.set_xlabel("Capital per effective worker, $k$")
#ax.set_ylabel("Investment per effective worker")
#ax.set_title("Finding the Steady State")
#ax.legend()
#ax.grid()
'''Plots for the actual Solow Swan model'''
#ax.plot(solution.t, solution.y[3], label = "Solow Swan Simulation")
#ax.set_xlabel("Time, $t$")
#ax.set_ylabel("Capital per Effective Worker, $k$")
#ax.set_title("ODE Simulation of the Solow Swan Model")
#ax.grid()
#ax.legend()
plt.show()
