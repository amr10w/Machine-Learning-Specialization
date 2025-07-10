import sympy

J,w,=sympy.symbols('J,w')

print(J)
print(w**2)
J=w**2
dj_dw=sympy.diff(J,w)
print(dj_dw)
dj_dw.subs([(w,2)])
print(dj_dw.subs([(w,2)]))

J,w=sympy.symbols('J,w')
print(J)
print(w)

J=1/(w**2)
print(J)
dj_dw=sympy.diff(J,w)
print(dj_dw)
print(dj_dw.subs([(w,4)]))

J=(2+3*w)**2
dj_dw = sympy.diff(J,w)
print(dj_dw)