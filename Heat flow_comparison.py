import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Library_term_project import *

import numpy as np
import matplotlib.pyplot as plt

#https://www.nature.com/articles/s41598-018-26108-x
L = 1
n = 10

T0= 33.5
T_source = 20.6
k=48
density=2650
capacity=1739
alpha = k/(density*capacity)

dx = L/n
nx=int(L/dx)+1

t_final = 24*60
dt = 1
nt=int(t_final/dt)

r = alpha * dt / (dx ** 2)  # Stability criterion

# Initial condition
T_initial = 20.6  # Initial temperature of the soil (°C)
T_soil = np.ones(nx) * T_initial

# Boundary conditions
T_left = 33.5  # Temperature at left boundary (°C)
# T_right = 0.0  # Temperature at right boundary (°C)

# Applying boundary conditions
T_soil[0] = T_left
T_soil[-1] = T_soil[-2]

# Explicit finite difference method
for t in range(1, nt):
    T_new = np.copy(T_soil)
    for i in range(1, nx - 1):
        T_new[i] = T_soil[i] + r * (T_soil[i+1] - 2 * T_soil[i] + T_soil[i-1])
    T_soil = np.copy(T_new)

# Plotting the temperature distribution
x = np.linspace(0, L, nx)
plt.plot(x, T_soil)
plt.scatter(x, T_soil)
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in Soil')
plt.grid(True)
plt.show()