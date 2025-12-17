import sympy as smp

# 1. Define Symbols
t, g = smp.symbols('t g')
m1, m2, l1, l2 = smp.symbols('m1 m2 l1 l2')
the1, the2 = smp.symbols(r'theta1 theta2', cls=smp.Function)
the1 = the1(t)
the2 = the2(t)

the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# 2. Define Physics (Kinematics & Energy)
x1 = l1*smp.sin(the1)
y1 = -l1*smp.cos(the1)
x2 = l1*smp.sin(the1)+l2*smp.sin(the2)
y2 = -l1*smp.cos(the1)-l2*smp.cos(the2)

T1 = 1/2*m1*(smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 1/2*m2*(smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
V1 = m1*g*y1
V2 = m2*g*y2
L = T1 + T2 - V1 - V2

# 3. Lagrange Equations
LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

# 4. Solve for accelerations
print("Solving system... (this may take a moment)")
sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=True, rational=False)

# 5. Output the results as strings
print("-" * 30)
print("PASTE THIS INTO YOUR WEB SCRIPT:")
print("-" * 30)
print(f"the1_dd_str = '{str(sols[the1_dd])}'")
print(f"the2_dd_str = '{str(sols[the2_dd])}'")
print("-" * 30)