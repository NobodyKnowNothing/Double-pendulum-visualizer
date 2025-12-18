import js
import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.embed_limit'] = 100
import matplotlib.pyplot as plt
from matplotlib import animation
import io
import asyncio

async def main():
    while True:
        # --- Wait for Data ---
        js.document.getElementById("status").innerText = "Status: Waiting for C++ data..."
        while not hasattr(js, 'cppData') or not hasattr(js, 'cppDataAdaptive'):
            await asyncio.sleep(0.1)

        # --- Setup Parameters & Constants ---
        _g, _m1, _m2 = 9.81, 2, 1
        l1_val, l2_val = 2, 1
        t_end_val = 40
        t_eval = np.linspace(0, t_end_val, 1001)

        js.document.getElementById("status").innerText = "Status: Processing Data..."

        data_cpp = np.genfromtxt(io.StringIO(js.cppData), delimiter=',', skip_header=1)
        t_cpp = data_cpp[:, 0]
        th1_cpp = data_cpp[:, 1]
        w1_cpp = data_cpp[:, 2]
        th2_cpp = data_cpp[:, 3]
        w2_cpp = data_cpp[:, 4]
        
        x1_cpp = l1_val * np.sin(th1_cpp)
        y1_cpp = -l1_val * np.cos(th1_cpp)
        x2_cpp = x1_cpp + l2_val * np.sin(th2_cpp)
        y2_cpp = y1_cpp - l2_val * np.cos(th2_cpp)

        # --- Load Adaptive C++ Data ---
        data_cpp_a = np.genfromtxt(io.StringIO(js.cppDataAdaptive), delimiter=',', skip_header=1)
        t_raw_a = data_cpp_a[:, 0]

        th1_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 1])
        th2_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 3])
        w1_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 2])
        w2_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 4])

        x1_cpp_a = l1_val * np.sin(th1_cpp_a)
        y1_cpp_a = -l1_val * np.cos(th1_cpp_a)
        x2_cpp_a = x1_cpp_a + l2_val * np.sin(th2_cpp_a)
        y2_cpp_a = y1_cpp_a - l2_val * np.cos(th2_cpp_a)

        # --- Setup Python Simulation ---
        js.document.getElementById("status").innerText = "Status: Solving Python ODE..."

        t_sym, g_sym = smp.symbols('t g')
        m1_sym, m2_sym, l1_sym, l2_sym = smp.symbols('m1 m2 l1 l2')
        the1, the2 = smp.symbols(r'theta1 theta2', cls=smp.Function)
        the1, the2 = the1(t_sym), the2(t_sym)
        
        eq1_str = '''(-1.0*g*m1*sin(theta1(t)) - 0.5*g*m2*sin(theta1(t) - 2*theta2(t)) - 0.5*g*m2*sin(theta1(t)) -
        0.5*l1*m2*sin(2*theta1(t) - 2*theta2(t))*Derivative(theta1(t), t)**2 - 1.0*l2*m2*sin(theta1(t) -
        theta2(t))*Derivative(theta2(t), t)**2)/(l1*(m1 - m2*cos(theta1(t) - theta2(t))**2 + m2))'''
        eq2_str = '''(0.5*g*m1*sin(2*theta1(t) - theta2(t)) - 0.5*g*m1*sin(theta2(t)) + 0.5*g*m2*sin(2*theta1(t) -
        theta2(t)) - 0.5*g*m2*sin(theta2(t)) + 1.0*l1*m1*sin(theta1(t) - theta2(t))*Derivative(theta1(t), t)**2 +
        1.0*l1*m2*sin(theta1(t) - theta2(t))*Derivative(theta1(t), t)**2 + 0.5*l2*m2*sin(2*theta1(t) -
        2*theta2(t))*Derivative(theta2(t), t)**2)/(l2*(m1 - m2*cos(theta1(t) - theta2(t))**2 + m2))'''

        z1, z2 = smp.symbols('z1 z2')
        dz1_expr = smp.sympify(eq1_str).subs({smp.diff(the1, t_sym): z1, smp.diff(the2, t_sym): z2})
        dz2_expr = smp.sympify(eq2_str).subs({smp.diff(the1, t_sym): z1, smp.diff(the2, t_sym): z2})

        dz1dt_f = smp.lambdify((t_sym, g_sym, m1_sym, m2_sym, l1_sym, l2_sym, the1, the2, z1, z2), dz1_expr)
        dz2dt_f = smp.lambdify((t_sym, g_sym, m1_sym, m2_sym, l1_sym, l2_sym, the1, the2, z1, z2), dz2_expr)

        def dSdt(S, t, g, m1, m2, L1, L2):
            th1_v, z1_v, th2_v, z2_v = S
            return [z1_v, dz1dt_f(t, g, m1, m2, L1, L2, th1_v, th2_v, z1_v, z2_v),
                    z2_v, dz2dt_f(t, g, m1, m2, L1, L2, th1_v, th2_v, z1_v, z2_v)]

        ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_eval, args=(_g, _m1, _m2, l1_val, l2_val))

        the1_py, w1_py = ans.T[0], ans.T[1]
        the2_py, w2_py = ans.T[2], ans.T[3]
        x1_py = l1_val * np.sin(the1_py)
        y1_py = -l1_val * np.cos(the1_py)
        x2_py = x1_py + l2_val * np.sin(the2_py)
        y2_py = y1_py - l2_val * np.cos(the2_py)

        # --- Energy ---
        def get_energy(th1, w1, th2, w2, m1, m2, l1, l2, g):
            V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
            T = 0.5 * m1 * (l1 * w1)**2 + 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2))
            return T + V

        energy_cpp = get_energy(th1_cpp, w1_cpp, th2_cpp, w2_cpp, _m1, _m2, l1_val, l2_val, _g)
        energy_py = get_energy(the1_py, w1_py, the2_py, w2_py, _m1, _m2, l1_val, l2_val, _g)
        energy_cpp_a = get_energy(th1_cpp_a, w1_cpp_a, th2_cpp_a, w2_cpp_a, _m1, _m2, l1_val, l2_val, _g)

        # --- Visualization ---
        plt.close('all') # Cleanup old figure
        fig, ax = plt.subplots(2, 1, figsize=(6, 10), facecolor='#111')
        plt.tight_layout(pad=4)
        
        ax[0].set_facecolor("k")
        ax[0].set_xlim(-4, 4); ax[0].set_ylim(-4, 4)
        ax[0].set_aspect('equal'); ax[0].axis('off')

        ax[1].set_facecolor("#111")
        ax[1].tick_params(colors='white')
        ax[1].xaxis.label.set_color('white'); ax[1].yaxis.label.set_color('white')
        ax[1].set_xlim(0, 40); ax[1].set_ylim(np.min(energy_cpp)-1, np.max(energy_cpp)+1)
        ax[1].set_ylabel("Total Energy"); ax[1].set_xlabel("Time (s)")
        ax[1].grid(True, alpha=0.3)

        ln_cpp, = ax[0].plot([], [], 'ro-', lw=3, markersize=8, label='C++ RK4')
        ln_py, = ax[0].plot([], [], 'c.--', lw=2, markersize=6, alpha=0.7, label='Python SciPy')
        ln_cpp_a, = ax[0].plot([], [], 'm-', lw=2, label='C++ Adaptive')
        
        trace_cpp, = ax[0].plot([], [], 'y-', lw=1, alpha=0.5)
        trace_py, = ax[0].plot([], [], 'b-', lw=1, alpha=0.5)
        trace_cpp_a, = ax[0].plot([], [], 'm-', lw=1, alpha=0.3)
        ax[0].legend(loc='upper right', facecolor='black', labelcolor='white')
        
        en_line_cpp, = ax[1].plot([], [], 'r-', label='C++ RK4')
        en_line_py, = ax[1].plot([], [], 'c--', label='Python SciPy')
        en_line_cpp_a, = ax[1].plot([], [], 'm-', label='C++ Adaptive')
        ax[1].legend(facecolor='black', labelcolor='white')

        trace_cpp_x, trace_cpp_y = [], []
        trace_py_x, trace_py_y = [], []
        max_trace = 50

        from pyscript import display

        # Animation loop
        min_len = min(len(t_cpp), len(t_eval)) 
        js.document.getElementById("status").innerText = "Status: Simulation Running";
        
        for i in range(0, min_len, 2):
            # Check if reset was clicked
            if not hasattr(js, 'cppData'):
                break

            ln_cpp.set_data([0, x1_cpp[i], x2_cpp[i]], [0, y1_cpp[i], y2_cpp[i]])
            trace_cpp_x.append(x2_cpp[i]); trace_cpp_y.append(y2_cpp[i])
            if len(trace_cpp_x) > max_trace: trace_cpp_x.pop(0); trace_cpp_y.pop(0)
            trace_cpp.set_data(trace_cpp_x, trace_cpp_y)

            ln_py.set_data([0, x1_py[i], x2_py[i]], [0, y1_py[i], y2_py[i]])
            trace_py_x.append(x2_py[i]); trace_py_y.append(y2_py[i])
            if len(trace_py_x) > max_trace: trace_py_x.pop(0); trace_py_y.pop(0)
            trace_py.set_data(trace_py_x, trace_py_y)

            ln_cpp_a.set_data([0, x1_cpp_a[i], x2_cpp_a[i]], [0, y1_cpp_a[i], y2_cpp_a[i]])
            trace_cpp_a.set_data(x2_cpp_a[max(0, i-50):i], y2_cpp_a[max(0, i-50):i])

            en_line_cpp.set_data(t_cpp[:i], energy_cpp[:i])
            en_line_py.set_data(t_eval[:i], energy_py[:i])
            en_line_cpp_a.set_data(t_eval[:i], energy_cpp_a[:i])

            display(fig, target="plot-div", append=False)
            await asyncio.sleep(0.01)

        if hasattr(js, 'cppData'):
            js.document.getElementById("status").innerText = "Status: Simulation Complete";

asyncio.ensure_future(main())
