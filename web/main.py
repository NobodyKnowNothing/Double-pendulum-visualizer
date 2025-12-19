import js
import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import io
import asyncio

async def reset_simulation(event):
    global is_running
    is_running = False # Signal current loop to stop
    
    # 1. Clear UI and show status
    status_el = js.document.getElementById("status")
    status_el.innerText = "Status: Stopping current simulation..."
    js.document.getElementById("plot-div").innerHTML = ""
    
    await asyncio.sleep(0.2) # Give the loop time to exit
    
    # 2. Tell JS to re-run C++
    status_el.innerText = "Status: Re-running C++ engine..."
    await js.runCppAndTransferFile()
    
    # 3. Restart the Python logic
    asyncio.ensure_future(main())

is_running = False
async def main():
    global is_running
    is_running = True # Set flag when starting
    # --- Setup Parameters & Constants ---
    _g, _m1, _m2 = 9.81, 2, 1
    l1_val, l2_val = 2, 1
    t_end_val = 40
    t_eval = np.linspace(0, t_end_val, 1001)

    # --- Load C++ Data ---
    while not hasattr(js, 'cppData') or not hasattr(js, 'cppDataAdaptive'):
        await asyncio.sleep(0.1)

    js.document.getElementById("status").innerText = "Status: Processing Data..."

    data_cpp = np.genfromtxt(io.StringIO(js.cppData), delimiter=',', skip_header=1)
    t_cpp = data_cpp[:, 0]
    th1_cpp = data_cpp[:, 1]
    w1_cpp = data_cpp[:, 2]
    th2_cpp = data_cpp[:, 3]
    w2_cpp = data_cpp[:, 4]
    # Extract Lyapunov log sums and compute exponents (λ = log_sum / t)
    l1_cpp_raw = data_cpp[:, 5]
    l2_cpp_raw = data_cpp[:, 6]
    l3_cpp_raw = data_cpp[:, 7]
    l4_cpp_raw = data_cpp[:, 8]
    # Avoid division by zero at t=0
    t_safe = np.where(t_cpp > 0, t_cpp, 1e-10)
    lyap1_cpp = l1_cpp_raw / t_safe
    lyap2_cpp = l2_cpp_raw / t_safe
    lyap3_cpp = l3_cpp_raw / t_safe
    lyap4_cpp = l4_cpp_raw / t_safe
    
    x1_cpp = l1_val * np.sin(th1_cpp)
    y1_cpp = -l1_val * np.cos(th1_cpp)
    x2_cpp = x1_cpp + l2_val * np.sin(th2_cpp)
    y2_cpp = y1_cpp - l2_val * np.cos(th2_cpp)

    # --- Load Adaptive C++ Data ---
    data_cpp_a = np.genfromtxt(io.StringIO(js.cppDataAdaptive), delimiter=',', skip_header=1)
    t_raw_a = data_cpp_a[:, 0]

    # Interpolate adaptive data to match the Python/Fixed-C++ time grid (t_eval)
    th1_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 1])
    th2_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 3])
    w1_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 2])
    w2_cpp_a = np.interp(t_eval, t_raw_a, data_cpp_a[:, 4])
    # Interpolate Lyapunov data for adaptive C++
    l1_cpp_a_raw = np.interp(t_eval, t_raw_a, data_cpp_a[:, 5])
    l2_cpp_a_raw = np.interp(t_eval, t_raw_a, data_cpp_a[:, 6])
    l3_cpp_a_raw = np.interp(t_eval, t_raw_a, data_cpp_a[:, 7])
    l4_cpp_a_raw = np.interp(t_eval, t_raw_a, data_cpp_a[:, 8])
    t_eval_safe = np.where(t_eval > 0, t_eval, 1e-10)
    lyap1_cpp_a = l1_cpp_a_raw / t_eval_safe
    lyap2_cpp_a = l2_cpp_a_raw / t_eval_safe
    lyap3_cpp_a = l3_cpp_a_raw / t_eval_safe
    lyap4_cpp_a = l4_cpp_a_raw / t_eval_safe

    # Calculate Cartesian coordinates
    x1_cpp_a = l1_val * np.sin(th1_cpp_a)
    y1_cpp_a = -l1_val * np.cos(th1_cpp_a)
    x2_cpp_a = x1_cpp_a + l2_val * np.sin(th2_cpp_a)
    y2_cpp_a = y1_cpp_a - l2_val * np.cos(th2_cpp_a)

    # --- Setup Python Simulation ---
    js.document.getElementById("status").innerText = "Status: Solving Python ODE..."

    # Define symbols and equations
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
        the1_v, z1_v, the2_v, z2_v = S
        return [z1_v, dz1dt_f(t, g, m1, m2, L1, L2, the1_v, the2_v, z1_v, z2_v),
                z2_v, dz2dt_f(t, g, m1, m2, L1, L2, the1_v, the2_v, z1_v, z2_v)]

    ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_eval, args=(_g, _m1, _m2, l1_val, l2_val))

    the1_py = ans.T[0]
    w1_py = ans.T[1]
    the2_py = ans.T[2]
    w2_py = ans.T[3]
    x1_py = l1_val * np.sin(the1_py)
    y1_py = -l1_val * np.cos(the1_py)
    x2_py = x1_py + l2_val * np.sin(the2_py)
    y2_py = y1_py - l2_val * np.cos(the2_py)

    # --- Lyapunov Exponent for Python LSODA ---
    # We solve the variational equations: dot(Phi) = J(S) * Phi
    js.document.getElementById("status").innerText = "Status: Computing Python Lyapunov Exponents (Variational)..."
    
    # 1. Derive Jacobian symbolicly for accuracy
    th1_s, th2_s, z1_s, z2_s = smp.symbols('th1_s th2_s z1_s z2_s')
    # Use dz1_expr/dz2_expr which already have Derivatives replaced by z1, z2
    expr1 = dz1_expr.subs({the1: th1_s, the2: th2_s, z1: z1_s, z2: z2_s})
    expr2 = dz2_expr.subs({the1: th1_s, the2: th2_s, z1: z1_s, z2: z2_s})
    
    state_vars = smp.Matrix([th1_s, z1_s, th2_s, z2_s])
    func_vec = smp.Matrix([z1_s, expr1, z2_s, expr2])
    Jac_mat = func_vec.jacobian(state_vars)
    Jac_f = smp.lambdify((th1_s, z1_s, th2_s, z2_s, g_sym, m1_sym, m2_sym, l1_sym, l2_sym), Jac_mat)

    # 2. Define combined ODE system
    def dVariationaldt(Y, t, g, m1, m2, l1, l2):
        state = Y[:4]
        Phi = Y[4:].reshape(4, 4)
        # Base dynamics
        dS = dSdt(state, t, g, m1, m2, l1, l2)
        # Jacobian at current state
        J = Jac_f(state[0], state[1], state[2], state[3], g, m1, m2, l1, l2)
        # dot(Phi) = J * Phi
        dPhi = np.dot(J, Phi)
        return np.concatenate([dS, dPhi.flatten()])

    # 3. Integrate with periodic orthonormalization
    Y = np.zeros(20)
    Y[:4] = [1, -3, -1, 5]  # Initial state
    Y[4:] = np.eye(4).flatten() # Initial perturbation matrix (Identity)
    
    log_sums = np.zeros(4)
    lyap_py_history = [np.zeros(4)]
    
    # We use the previous solution 'ans' to avoid redundant integration of the base state?
    # Actually it's easier to just integrate the combined system for consistency.
    for i in range(1, len(t_eval)):
        times = [t_eval[i-1], t_eval[i]]
        sol = odeint(dVariationaldt, Y, times, args=(_g, _m1, _m2, l1_val, l2_val))
        Y = sol[1]
        
        # Gram-Schmidt Orthonormalization (using QR decomposition)
        Phi = Y[4:].reshape(4, 4)
        # QR of the columns of Phi
        Q, R = np.linalg.qr(Phi)
        
        # Accumulate logarithms of the scales (diagonal of R)
        # We take abs because orientations can flip
        log_sums += np.log(np.abs(np.diag(R)))
        
        # Reset Phi to the orthonormal basis Q
        Y[4:] = Q.flatten()
        
        # Current Lyapunov Exponents: λᵢ = Σ log(scaleᵢ) / t
        t_now = t_eval[i]
        lyap_py_history.append(log_sums / t_now)
    
    lyap_py_arr = np.array(lyap_py_history)
    lyap1_py = lyap_py_arr[:, 0]
    lyap2_py = lyap_py_arr[:, 1]
    lyap3_py = lyap_py_arr[:, 2]
    lyap4_py = lyap_py_arr[:, 3]

    # --- Energy ---
    def get_energy(th1, w1, th2, w2, m1, m2, l1, l2, g):
        V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        T = 0.5 * m1 * (l1 * w1)**2 + 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2))
        return T + V

    energy_cpp = get_energy(th1_cpp, w1_cpp, th2_cpp, w2_cpp, _m1, _m2, l1_val, l2_val, _g)
    energy_py = get_energy(the1_py, w1_py, the2_py, w2_py, _m1, _m2, l1_val, l2_val, _g)
    energy_cpp_a = get_energy(th1_cpp_a, w1_cpp_a, th2_cpp_a, w2_cpp_a, _m1, _m2, l1_val, l2_val, _g)

    # --- Visualization ---
    js.document.getElementById("status").innerText = "Status: Starting Live Animation..."

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), facecolor='#111')
    plt.tight_layout(pad=3.0)
    
    # Pendulum animation subplot
    ax[0].set_facecolor("k")
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].set_aspect('equal')
    ax[0].axis('off')

    # Energy subplot
    ax[1].set_facecolor("#111")
    ax[1].tick_params(colors='white')
    ax[1].xaxis.label.set_color('white')
    ax[1].yaxis.label.set_color('white')
    ax[1].set_xlim(0, 40)
    ax[1].set_ylim(np.min(energy_cpp)-1, np.max(energy_cpp)+1)
    ax[1].set_ylabel("Total Energy")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.3)

    # Lyapunov exponent subplot
    ax[2].set_facecolor("#111")
    ax[2].tick_params(colors='white')
    ax[2].xaxis.label.set_color('white')
    ax[2].yaxis.label.set_color('white')
    ax[2].set_xlim(0, 40)
    # Set y-limits based on all Lyapunov data
    lyap_all = np.concatenate([lyap1_cpp[1:], lyap1_cpp_a[1:], lyap1_py[1:]])
    y_min = max(np.min(lyap_all), -10)
    y_max = min(np.max(lyap_all), 50)
    ax[2].set_ylim(y_min - 0.5, y_max + 1)
    
    import matplotlib.ticker as ticker
    ax[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=15)) # Dynamic but dense ticks
    ax[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    ax[2].set_ylabel("Lyapunov Exponent λ₁")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)

    # Pendulum lines
    ln_cpp, = ax[0].plot([], [], 'ro-', lw=3, markersize=8, label='RK4 Rigid Step Size in C++')
    ln_py, = ax[0].plot([], [], 'c.--', lw=2, markersize=6, alpha=0.7, label='Adaptive-Step LSODA using SciPy in Python')
    ln_cpp_a, = ax[0].plot([], [], 'm-', lw=2, label='RK4 Adaptive Step Size in C++')
    
    trace_cpp, = ax[0].plot([], [], 'y-', lw=1, alpha=0.5)
    trace_py, = ax[0].plot([], [], 'b-', lw=1, alpha=0.5)
    trace_cpp_a, = ax[0].plot([], [], 'm-', lw=1, alpha=0.3)
    ax[0].legend(loc='upper right', facecolor='black', labelcolor='white', fontsize=6)
    
    # Energy lines
    en_line_cpp, = ax[1].plot([], [], 'r-', label='RK4 (C++)')
    en_line_py, = ax[1].plot([], [], 'c--', label='LSODA (Python)')
    en_line_cpp_a, = ax[1].plot([], [], 'm-', label='Adaptive RK4 (C++)')
    ax[1].legend(facecolor='black', labelcolor='white', fontsize=7)

    # Lyapunov lines
    lyap_line_cpp, = ax[2].plot([], [], 'r-', label='RK4 (C++)')
    lyap_line_py, = ax[2].plot([], [], 'c--', label='LSODA (Python)')
    lyap_line_cpp_a, = ax[2].plot([], [], 'm-', label='Adaptive RK4 (C++)')
    ax[2].legend(facecolor='black', labelcolor='white', fontsize=7)

    # Animation Logic
    trace_cpp_x, trace_cpp_y = [], []
    trace_py_x, trace_py_y = [], []
    max_trace = 50

    from pyscript import display

    async def run_animation():
        min_len = min(len(t_cpp), len(t_eval)) 
        for i in range(0, min_len, 2):
            if not is_running: # Check flag every frame
                return
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

            # Adaptive Update
            ln_cpp_a.set_data([0, x1_cpp_a[i], x2_cpp_a[i]], [0, y1_cpp_a[i], y2_cpp_a[i]])
            trace_cpp_a.set_data(x2_cpp_a[max(0, i-50):i], y2_cpp_a[max(0, i-50):i])

            # Energy Plot
            en_line_cpp.set_data(t_cpp[:i], energy_cpp[:i])
            en_line_py.set_data(t_eval[:i], energy_py[:i])
            en_line_cpp_a.set_data(t_eval[:i], energy_cpp_a[:i])

            # Lyapunov Plot (show all including initial noise)
            lyap_line_cpp.set_data(t_cpp[1:i], lyap1_cpp[1:i])
            lyap_line_py.set_data(t_eval[1:i], lyap1_py[1:i])
            lyap_line_cpp_a.set_data(t_eval[1:i], lyap1_cpp_a[1:i])

            # Display the updated figure
            display(fig, target="plot-div", append=False)
            
            # Control playback speed (yielding to bridge)
            await asyncio.sleep(0.01)

    js.document.getElementById("status").innerText = "Status: Simulation Running";
    await run_animation()
    js.document.getElementById("status").innerText = "Status: Simulation Complete";

asyncio.ensure_future(main())
