import js
import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['animation.embed_limit'] = 50.0
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import io
import asyncio

async def main():
    # --- Load C++ Data ---
    while not hasattr(js, 'cppData'):
        await asyncio.sleep(0.1)

    data_cpp = pd.read_csv(io.StringIO(js.cppData))
    t_cpp = data_cpp['t'].values
    th1_cpp = data_cpp['th1'].values
    th1_cpp = data_cpp['th1'].values
    th2_cpp = data_cpp['th2'].values
    
    l1_val, l2_val = 2, 1
    x1_cpp = l1_val * np.sin(th1_cpp)
    y1_cpp = -l1_val * np.cos(th1_cpp)
    x2_cpp = x1_cpp + l2_val * np.sin(th2_cpp)
    y2_cpp = y1_cpp - l2_val * np.cos(th2_cpp)

    # --- Setup Python Simulation ---

    # Define symbols again so SymPy knows what 'theta1(t)' etc mean in the strings
    t, g = smp.symbols('t g')
    m1, m2, l1, l2 = smp.symbols('m1 m2 l1 l2')
    the1, the2 = smp.symbols(r'theta1 theta2', cls=smp.Function)
    the1, the2 = the1(t), the2(t)
    the1_d, the2_d = smp.diff(the1, t), smp.diff(the2, t)

    eq1_str = '''(-1.0*g*m1*sin(theta1(t)) - 0.5*g*m2*sin(theta1(t) - 2*theta2(t)) - 0.5*g*m2*sin(theta1(t)) -
    0.5*l1*m2*sin(2*theta1(t) - 2*theta2(t))*Derivative(theta1(t), t)**2 - 1.0*l2*m2*sin(theta1(t) -
    theta2(t))*Derivative(theta2(t), t)**2)/(l1*(m1 - m2*cos(theta1(t) - theta2(t))**2 + m2))'''
    eq2_str = '''(0.5*g*m1*sin(2*theta1(t) - theta2(t)) - 0.5*g*m1*sin(theta2(t)) + 0.5*g*m2*sin(2*theta1(t) -
    theta2(t)) - 0.5*g*m2*sin(theta2(t)) + 1.0*l1*m1*sin(theta1(t) - theta2(t))*Derivative(theta1(t), t)**2 +
    1.0*l1*m2*sin(theta1(t) - theta2(t))*Derivative(theta1(t), t)**2 + 0.5*l2*m2*sin(2*theta1(t) -
    2*theta2(t))*Derivative(theta2(t), t)**2)/(l2*(m1 - m2*cos(theta1(t) - theta2(t))**2 + m2))'''

    # Convert strings back to SymPy expressions instantly
    dz1_expr = smp.sympify(eq1_str)
    dz2_expr = smp.sympify(eq2_str)


    # Create numerical functions (lambdify)
    # Note: sympify might treat 'Derivative(theta1, t)' as a symbol.
    # We map the specific derivative symbols to simple variables 'z1' and 'z2' for lambdify.
    z1, z2 = smp.symbols('z1 z2')

    # Replace derivatives in expression with z1/z2 so lambdify works cleanly
    dz1_expr = dz1_expr.subs({smp.diff(the1, t): z1, smp.diff(the2, t): z2})
    dz2_expr = dz2_expr.subs({smp.diff(the1, t): z1, smp.diff(the2, t): z2})

    # Lambdify
    dz1dt_f = smp.lambdify((t, g, m1, m2, l1, l2, the1, the2, z1, z2), dz1_expr)
    dz2dt_f = smp.lambdify((t, g, m1, m2, l1, l2, the1, the2, z1, z2), dz2_expr)

    def dSdt(S, t, g, m1, m2, L1, L2):
        the1_v, z1_v, the2_v, z2_v = S
        return [
        z1_v,
        dz1dt_f(t, g, m1, m2, L1, L2, the1_v, the2_v, z1_v, z2_v),
        z2_v,
        dz2dt_f(t, g, m1, m2, L1, L2, the1_v, the2_v, z1_v, z2_v)
        ]

    # Run Simulation
    t_eval = np.linspace(0, 40, 1001)
    ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_eval, args=(9.81, 2, 1, 2, 1))

    # Extract results
    the1_py = ans.T[0]
    the2_py = ans.T[2]
    l1_val, l2_val = 2, 1
    x1_py = l1_val * np.sin(the1_py)
    y1_py = -l1_val * np.cos(the1_py)
    x2_py = x1_py + l2_val * np.sin(the2_py)
    y2_py = y1_py - l2_val * np.cos(the2_py)

    # --- Visualization ---
    js.document.getElementById("status").innerText = "Status: Rendering Animation..."

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("k")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Plot elements
    ln_cpp, = ax.plot([], [], 'ro-', lw=3, markersize=8, label='C++ RK4')
    ln_py, = ax.plot([], [], 'c.--', lw=2, markersize=6, alpha=0.7, label='Python SciPy')
    trace_cpp, = ax.plot([], [], 'y-', lw=1, alpha=0.5)
    trace_py, = ax.plot([], [], 'b-', lw=1, alpha=0.5)
    ax.legend(loc='upper right')

    # Animation Logic
    trace_cpp_x, trace_cpp_y = [], []
    trace_py_x, trace_py_y = [], []
    max_trace = 50

    step = 1
    min_len = min(len(t_cpp), len(t_eval)) 
    frames = range(0, min_len, step)

    def animate(i):
        # C++ Update
        ln_cpp.set_data([0, x1_cpp[i], x2_cpp[i]], [0, y1_cpp[i], y2_cpp[i]])
        trace_cpp_x.append(x2_cpp[i])
        trace_cpp_y.append(y2_cpp[i])
        if len(trace_cpp_x) > max_trace: trace_cpp_x.pop(0); trace_cpp_y.pop(0)
        trace_cpp.set_data(trace_cpp_x, trace_cpp_y)

        # Python Update
        ln_py.set_data([0, x1_py[i], x2_py[i]], [0, y1_py[i], y2_py[i]])
        trace_py_x.append(x2_py[i])
        trace_py_y.append(y2_py[i])
        if len(trace_py_x) > max_trace: trace_py_x.pop(0); trace_py_y.pop(0)
        trace_py.set_data(trace_py_x, trace_py_y)

        return ln_cpp, ln_py, trace_cpp, trace_py

    # Render to HTML JS format (Best for browser compatibility)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=40, blit=False)
    html_video = ani.to_jshtml()

    # Inject into DOM
    # Inject into DOM
    plot_div = js.document.getElementById("plot-div")
    plot_div.innerHTML = html_video
    
    # Force execution of scripts embedded in the HTML (innerHTML doesn't run them)
    scripts = plot_div.getElementsByTagName("script")
    # Create a static list because the collection is live
    script_list = [scripts.item(i) for i in range(scripts.length)]
    
    for old_script in script_list:
        new_script = js.document.createElement("script")
        if old_script.src:
            new_script.src = old_script.src
        if old_script.text:
            new_script.text = old_script.text
        if old_script.type:
            new_script.type = old_script.type
        old_script.parentNode.replaceChild(new_script, old_script)

    js.document.getElementById("status").innerText = "Status: Simulation Complete";

asyncio.ensure_future(main())