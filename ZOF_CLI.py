#!/usr/bin/env python3
"""
ZOF_CLI.py
Command-line Zero-Of-Function solver implementing:
1. Bisection
2. Regula Falsi (False Position)
3. Secant
4. Newton-Raphson
5. Fixed Point Iteration
6. Modified Secant

Dependencies: sympy, numpy
"""

import sys
import math
from sympy import sympify, symbols, lambdify, diff
import numpy as np

x = symbols('x')

def make_function(expr_str):
    """Return a Python callable f(x) from expression string."""
    expr = sympify(expr_str)
    f = lambdify(x, expr, "numpy")
    return f, expr

def print_table_header(method_name):
    print("\nMethod:", method_name)
    print("-" * 80)
    print(f"{'iter':>4} | {'a':>12} | {'b/xn':>12} | {'f(a)':>12} | {'f(b)/f(xn)':>12} | {'error':>10}")
    print("-" * 80)

def bisection(f, a, b, tol=1e-6, maxiter=50):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for bisection.")
    logs = []
    print_table_header("Bisection")
    c = a
    for i in range(1, maxiter + 1):
        c_prev = c
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(c - c_prev) if i > 1 else None
        logs.append((i, a, b, fa, fb, fc, error))
        print(f"{i:4d} | {a:12.6g} | {b:12.6g} | {fa:12.6g} | {fb:12.6g} | {error if error is not None else '---':>10}")
        if abs(fc) <= tol or (error is not None and error <= tol):
            return c, abs(fc), i, logs
        # choose subinterval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(f(c)), maxiter, logs

def regula_falsi(f, a, b, tol=1e-6, maxiter=50):
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    logs = []
    print_table_header("Regula Falsi (False Position)")
    c = a
    for i in range(1, maxiter + 1):
        c_prev = c
        c = (a * fb - b * fa) / (fb - fa)  # false position
        fc = f(c)
        error = abs(c - c_prev) if i > 1 else None
        logs.append((i, a, b, fa, fb, fc, error))
        print(f"{i:4d} | {a:12.6g} | {b:12.6g} | {fa:12.6g} | {fb:12.6g} | {error if error is not None else '---':>10}")
        if abs(fc) <= tol or (error is not None and error <= tol):
            return c, abs(fc), i, logs
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(f(c)), maxiter, logs

def secant(f, x0, x1, tol=1e-6, maxiter=50):
    logs = []
    print_table_header("Secant")
    for i in range(1, maxiter + 1):
        f0 = f(x0); f1 = f(x1)
        if abs(f1 - f0) < 1e-15:
            raise ZeroDivisionError("Denominator near zero in secant method.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        logs.append((i, x0, x1, f0, f1, x2, error))
        print(f"{i:4d} | {x0:12.6g} | {x1:12.6g} | {f0:12.6g} | {f1:12.6g} | {error:10.6g}")
        if error <= tol or abs(f(x2)) <= tol:
            return x2, abs(f(x2)), i, logs
        x0, x1 = x1, x2
    return x2, abs(f(x2)), maxiter, logs

def newton_raphson(f, fprime, x0, tol=1e-6, maxiter=50):
    logs = []
    print_table_header("Newton-Raphson")
    xk = x0
    for i in range(1, maxiter + 1):
        fx = f(xk); fpx = fprime(xk)
        if abs(fpx) < 1e-15:
            raise ZeroDivisionError("Derivative near zero in Newton-Raphson.")
        x_next = xk - fx / fpx
        error = abs(x_next - xk)
        logs.append((i, xk, fx, fpx, x_next, error))
        print(f"{i:4d} | {xk:12.6g} | {fx:12.6g} | {fpx:12.6g} | {error:10.6g}")
        if error <= tol or abs(f(x_next)) <= tol:
            return x_next, abs(f(x_next)), i, logs
        xk = x_next
    return xk, abs(f(xk)), maxiter, logs

def fixed_point_iteration(g_func, x0, tol=1e-6, maxiter=50):
    logs = []
    print_table_header("Fixed Point Iteration (x = g(x))")
    xk = x0
    for i in range(1, maxiter + 1):
        x_next = g_func(xk)
        error = abs(x_next - xk)
        logs.append((i, xk, x_next, error))
        print(f"{i:4d} | {xk:12.6g} | {x_next:12.6g} | {'---':12} | {error:10.6g}")
        if error <= tol:
            return x_next, abs(error), i, logs
        xk = x_next
    return xk, abs(error), maxiter, logs

def modified_secant(f, x0, delta=1e-4, tol=1e-6, maxiter=50):
    logs = []
    print_table_header("Modified Secant")
    xk = x0
    for i in range(1, maxiter + 1):
        fx = f(xk)
        denom = f(xk + delta * xk) - fx
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Denominator near zero in Modified Secant.")
        x_next = xk - (delta * xk * fx) / denom
        error = abs(x_next - xk)
        logs.append((i, xk, fx, x_next, error))
        print(f"{i:4d} | {xk:12.6g} | {fx:12.6g} | {'---':12} | {error:10.6g}")
        if error <= tol or abs(f(x_next)) <= tol:
            return x_next, abs(f(x_next)), i, logs
        xk = x_next
    return xk, abs(f(xk)), maxiter, logs

def run_cli():
    print("Zero-Of-Function (ZOF) Solver - CLI")
    print("Enter the function in variable x (e.g. x**3 - x - 2)")
    f_str = input("f(x) = ").strip()
    f, expr = make_function(f_str)

    print("\nChoose method:")
    print("1. Bisection")
    print("2. Regula Falsi (False Position)")
    print("3. Secant")
    print("4. Newton-Raphson")
    print("5. Fixed Point Iteration")
    print("6. Modified Secant")

    choice = input("Method (1-6): ").strip()
    tol = float(input("Tolerance (e.g. 1e-6): ") or 1e-6)
    maxiter = int(input("Max iterations (e.g. 50): ") or 50)

    try:
        if choice == "1":
            a = float(input("Left endpoint a: "))
            b = float(input("Right endpoint b: "))
            root, final_err, iters, logs = bisection(f, a, b, tol, maxiter)
        elif choice == "2":
            a = float(input("Left endpoint a: "))
            b = float(input("Right endpoint b: "))
            root, final_err, iters, logs = regula_falsi(f, a, b, tol, maxiter)
        elif choice == "3":
            x0 = float(input("Initial guess x0: "))
            x1 = float(input("Initial guess x1: "))
            root, final_err, iters, logs = secant(f, x0, x1, tol, maxiter)
        elif choice == "4":
            x0 = float(input("Initial guess x0: "))
            # create derivative
            expr = sympify(f_str)
            fprime_expr = diff(expr, x)
            fprime = lambdify(x, fprime_expr, "numpy")
            root, final_err, iters, logs = newton_raphson(f, fprime, x0, tol, maxiter)
        elif choice == "5":
            print("Fixed point needs g(x) such that x = g(x).")
            g_str = input("g(x) = ").strip()
            g_func, _ = make_function(g_str)
            x0 = float(input("Initial guess x0: "))
            root, final_err, iters, logs = fixed_point_iteration(g_func, x0, tol, maxiter)
        elif choice == "6":
            x0 = float(input("Initial guess x0: "))
            delta = float(input("Delta for modified secant (e.g. 1e-4): ") or 1e-4)
            root, final_err, iters, logs = modified_secant(f, x0, delta, tol, maxiter)
        else:
            print("Invalid choice.")
            return
    except Exception as e:
        print("Error during execution:", e)
        return

    print("\nFinal estimated root:", root)
    print("Final function value (or error):", final_err)
    print("Iterations used:", iters)
    print("-" * 40)
    print("Done.")

if __name__ == "__main__":
    run_cli()
