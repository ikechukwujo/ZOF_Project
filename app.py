from flask import Flask, render_template, request, redirect, url_for
from sympy import sympify, symbols, lambdify, diff
import numpy as np
import math

app = Flask(__name__)
x = symbols('x')

# --- Helper: build function from string
def make_function(expr_str):
    expr = sympify(expr_str)
    f = lambdify(x, expr, "numpy")
    return f, expr

# We'll include simpler wrappers that return logs as lists of dicts for rendering.
def bisection_logs(f, a, b, tol=1e-6, maxiter=50):
    fa = f(a); fb = f(b)
    logs = []
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    c = a
    for i in range(1, maxiter + 1):
        c_prev = c
        c = (a + b) / 2.0
        fc = float(f(c))
        error = abs(c - c_prev) if i > 1 else None
        logs.append({'iter': i, 'a': a, 'b': b, 'fa': float(fa), 'fb': float(fb), 'fc': fc, 'error': error})
        if abs(fc) <= tol or (error is not None and error <= tol):
            return c, abs(fc), i, logs
        if float(fa) * float(fc) < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(float(f(c))), maxiter, logs

def regula_falsi_logs(f, a, b, tol=1e-6, maxiter=50):
    fa = f(a); fb = f(b)
    logs = []
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    c = a
    for i in range(1, maxiter + 1):
        c_prev = c
        c = (a * fb - b * fa) / (fb - fa)
        fc = float(f(c))
        error = abs(c - c_prev) if i > 1 else None
        logs.append({'iter': i, 'a': a, 'b': b, 'fa': float(fa), 'fb': float(fb), 'fc': fc, 'error': error})
        if abs(fc) <= tol or (error is not None and error <= tol):
            return c, abs(fc), i, logs
        if float(fa) * float(fc) < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return c, abs(float(f(c))), maxiter, logs

def secant_logs(f, x0, x1, tol=1e-6, maxiter=50):
    logs = []
    for i in range(1, maxiter + 1):
        f0 = float(f(x0)); f1 = float(f(x1))
        if abs(f1 - f0) < 1e-15:
            raise ZeroDivisionError("Denominator near zero.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        logs.append({'iter': i, 'x0': x0, 'x1': x1, 'f0': f0, 'f1': f1, 'x2': x2, 'error': error})
        if error <= tol or abs(float(f(x2))) <= tol:
            return x2, abs(float(f(x2))), i, logs
        x0, x1 = x1, x2
    return x2, abs(float(f(x2))), maxiter, logs

def newton_logs(f, fprime, x0, tol=1e-6, maxiter=50):
    logs = []
    xk = x0
    for i in range(1, maxiter + 1):
        fx = float(f(xk)); fpx = float(fprime(xk))
        if abs(fpx) < 1e-15:
            raise ZeroDivisionError("Derivative near zero.")
        x_next = xk - fx / fpx
        error = abs(x_next - xk)
        logs.append({'iter': i, 'xk': xk, 'fx': fx, 'fpx': fpx, 'x_next': x_next, 'error': error})
        if error <= tol or abs(float(f(x_next))) <= tol:
            return x_next, abs(float(f(x_next))), i, logs
        xk = x_next
    return xk, abs(float(f(xk))), maxiter, logs

def fixed_point_logs(g_func, x0, tol=1e-6, maxiter=50):
    logs = []
    xk = x0
    for i in range(1, maxiter + 1):
        x_next = float(g_func(xk))
        error = abs(x_next - xk)
        logs.append({'iter': i, 'xk': xk, 'x_next': x_next, 'error': error})
        if error <= tol:
            return x_next, error, i, logs
        xk = x_next
    return xk, error, maxiter, logs

def modified_secant_logs(f, x0, delta=1e-4, tol=1e-6, maxiter=50):
    logs = []
    xk = x0
    for i in range(1, maxiter + 1):
        fx = float(f(xk))
        denom = float(f(xk + delta * xk) - fx)
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Denominator near zero.")
        x_next = xk - (delta * xk * fx) / denom
        error = abs(x_next - xk)
        logs.append({'iter': i, 'xk': xk, 'fx': fx, 'x_next': x_next, 'error': error})
        if error <= tol or abs(float(f(x_next))) <= tol:
            return x_next, abs(float(f(x_next))), i, logs
        xk = x_next
    return xk, abs(float(f(xk))), maxiter, logs

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_msg = None
    if request.method == "POST":
        try:
            method = request.form['method']
            f_str = request.form['fexpr']
            tol = float(request.form.get('tol') or 1e-6)
            maxiter = int(request.form.get('maxiter') or 50)
            f, expr = make_function(f_str)

            if method in ('bisection', 'regula'):
                a = float(request.form['a']); b = float(request.form['b'])
                if method == 'bisection':
                    root, final_err, iters, logs = bisection_logs(f, a, b, tol, maxiter)
                else:
                    root, final_err, iters, logs = regula_falsi_logs(f, a, b, tol, maxiter)
            elif method == 'secant':
                x0 = float(request.form['x0']); x1 = float(request.form['x1'])
                root, final_err, iters, logs = secant_logs(f, x0, x1, tol, maxiter)
            elif method == 'newton':
                x0 = float(request.form['x0'])
                fprime_expr = diff(expr, x)
                fprime = lambdify(x, fprime_expr, "numpy")
                root, final_err, iters, logs = newton_logs(f, fprime, x0, tol, maxiter)
            elif method == 'fixed':
                g_str = request.form['gexpr']
                g_func, _ = make_function(g_str)
                x0 = float(request.form['x0'])
                root, final_err, iters, logs = fixed_point_logs(g_func, x0, tol, maxiter)
            elif method == 'modified':
                x0 = float(request.form['x0'])
                delta = float(request.form.get('delta') or 1e-4)
                root, final_err, iters, logs = modified_secant_logs(f, x0, delta, tol, maxiter)
            else:
                raise ValueError("Unknown method")
            result = {
                'root': root,
                'final_err': final_err,
                'iters': iters,
                'logs': logs,
                'method': method
            }
        except Exception as e:
            error_msg = str(e)

    return render_template("index.html", result=result, error_msg=error_msg)

if __name__ == "__main__":
    app.run(debug=True)
