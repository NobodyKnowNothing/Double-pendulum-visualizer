import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

t, g = smp.symbols('t g')
m1, m2, l1, l2 = smp.symbols('m1 m2 l1 l2')
theta1, theta2 = smp.symbols('theta1 theta2')
the1, the2 = smp.symbols(r'theta_1, theta_2', cls=smp.Function)

the1 = the1(t)
the2 = the2(t)

the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

x1 = l1*smp.sin(the1)
y1 = -l1*smp.cos(the1)
x2 = l1*smp.sin(the1)+l2*smp.sin(the2) #Check
y2 = -l1*smp.cos(the1)-l2*smp.cos(the2) #Check

#Kinetic Energy
T1 = 1/2*m1*(smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2*m2*(smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1 + T2

#Potential Energy
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2

#Lagrangian
L = T-V

#Lagrange's Equations
LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False) # Solves the Lagrange's Equations using linear double derivatives

# print(sols[the1_dd])
# print(sols[the2_dd])

# Convert 2 second order ODEs to a system of 4 first order ODEs, formats for a numerical python solver
dz1dt_f = smp.lambdify((t, g, m1, m2, l1, l2, the1, the2, the1_d, the2_d), sols[the1_dd])
dz2dt_f = smp.lambdify((t, g, m1, m2, l1, l2, the1, the2, the1_d, the2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

# When solving a system of ODEs, we define a vector S that contains all the variables we are solving for
# We define a function dSdt that returns the derivatives of S
# dSdt = [the1, z1, theta1, z2]
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2)
    ]


# Init cond
t_eval = np.linspace(0, 40, 1001)
g_val = 9.81
m1_val = 2
m2_val = 1
l1_val = 2
l2_val = 1

def simulate_python():
    ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_eval, args=(g_val, m1_val, m2_val, l1_val, l2_val))
    
    the1 = ans.T[0]
    the2 = ans.T[2]
    
    x1, y1, x2, y2 = get_x1y1x2y2(t_eval, the1, the2, l1_val, l2_val)
    return t_eval, x1, y1, x2, y2

def get_x1y1x2y2(t, the1, the2, L1, L2):
    return [
        L1*np.sin(the1),
        -L1*np.cos(the1),
        L1*np.sin(the1) + L2*np.sin(the2),
        -L1*np.cos(the1) - L2*np.cos(the2)
    ]

if __name__ == "__main__":
    t_data, x1, y1, x2, y2 = simulate_python()
    
    def animate(i):
        ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        return ln1,

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.set_facecolor("k")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ln1, = plt.plot([], [], "ro--", lw=3, markersize=8)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ani = animation.FuncAnimation(fig, animate, frames=len(t_data), interval=40)
    # Using 40ms interval to match real-time (40s / 1000 frames = 0.04s)
    
    print("Saving animation...")
    # ani.save("pen.gif", writer=PillowWriter(fps=25))
    plt.show()
