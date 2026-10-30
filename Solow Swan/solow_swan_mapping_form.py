#Solow Swan Model
import numpy as np
import scipy
import matplotlib.pyplot as plt

'''
Fixed Parameters
A - the total factor productivity
n - the population growth rate per **SET TIME**. Initially we can set it to 1 for no growth
s - the share of capital dedicated to investment, 0 < s < 1
delta - the capital depreciation rate per **SET TIME**, delta > 0
'''
A = 2
n = 1
g = 1
s = 0.3
delta = 0.4
alpha = 0.3
'''
System Equations
Initial model considers the population L as constant - other models consider population growing over time
If population growth is constant (dLdt/L = n), then 
K - the aggregate capital in the system, with initial value K0
We can also consider k - the aggregate capital per worker, K/L
The output of work, Y is modelled using the Cobb Douglas production function
'''
K0 = 11400000
L = 150000
def cobb_douglas(K, L, A, alpha):
    '''
    Cobb-Douglas acts as the production function linking capital, labour and output
    Inputs - capital K, Labour size L, total factor productivity A, share of output from capital alpha
    Output - the output of production (mathematically symbolised by Y) 
    '''
    return A * (K**alpha) * (L**(1-alpha))

def K_new(K, L, A, s, alpha, delta):
    '''
    Finds the next iteration's value of capital
    Inputs - current capital K, labour size L, share for investment s, share of output from capital alpha, capital depreciation rate delta
    Output - the next iteration's value of capital, K_(t+1)
    '''
    return s * cobb_douglas(K, L, A, alpha) + K * (1 - delta)

def k_func(k, A, s, alpha, delta):
    '''
    Considers the discrete mapping of capital per effective worker
    Inputs - the current capital per effective worker, k; total factor productivity, A; share for investment,
    the contribution to output from capital alpha and capital depreciation rate delta
    Output - the next value of capital per effective worker'''
    return A * s * k ** alpha + k*(1-delta)

k_star = ((s*A)/delta) ** (1/(1-alpha))
x_grid = np.linspace(0, int((k_star+2)), 200)
k_vals = k_func(x_grid, A, s, alpha, delta)
init_values = np.array([-1, 0.25, 0.5, 1, 2, 5])
output_values = np.zeros(200)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x_grid, k_vals, color = "red")
ax[0].plot(x_grid, x_grid, color = "blue")
ax[0].scatter(k_star, k_star, label = f"Steady state, k = {k_star}")
ax[0].set_xlabel("$k_t$")
ax[0].set_ylabel("$k_{t+1}$")
ax[0].set_title("Solow Swan Model")
for j in init_values:
    output_values[0] = j
    for t in range(1, 200):
        output_values[t] = k_func(output_values[t-1], A, s, alpha, delta)
    ax[1].plot(x_grid, output_values, label = f"$k_0$ = {j}")
ax[1].legend()
ax[1].set_xlabel("$k_t$")
ax[1].set_ylabel("$k_{t+1}$")
ax[1].set_title("Varying the initial condition for the Solow Swan Model")
plt.show()