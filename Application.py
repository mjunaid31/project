import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Library_term_project import *

import numpy as np
import matplotlib.pyplot as plt

alpha=0.02186
Tamp=32
t_now=20
t_shift=37
depth=1000
T_mean=9.5

def T(z):
    return T_mean-Tamp*np.exp(-z*(np.pi/365/alpha)**(1/2))*np.cos(2*np.pi/365*(t_now-t_shift-depth/2*(365/np.pi/alpha)**(1/2)))

zlist=[i for i in range(1000)]
Tlist=[T(z) for z in range(len(zlist))]



#Boundary conditions
# zlist[0]=0
# zlist[-1]=1000
# Tlist[0]=43
# Tlist[-1]=17.5


#plotting temperature versus depth
plt.plot(zlist,Tlist)
plt.xlabel('Depth (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in Soil')
plt.grid(True)
plt.show()





