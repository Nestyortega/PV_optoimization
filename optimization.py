import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def optimize_portfolio(num_intervals, DAM_price, panel_forecasting, battery_charge_last_day, battery_capacity):

    num_intervals = 24*4

    num_variables = num_intervals*3  # 24 hours, 4 15-minute intervals, 3 variables types
    num_constrains = num_intervals + 3  # constraints

    # Objective function coefficients (for maximization, flip signs)
    A = DAM_price # Market price of electricity in 15 minute intervals
    B = panel_forecasting # Panel output in 15 minute intervals
    C = battery_charge_last_day # Battery charge from previous day
    C1 = battery_capacity # Battery charge cappacity

    c = np.concatenate((A, A, A))

    # Inequality constraint coefficients (Ax <= b)

    A_ = np.zeros((num_constrains, num_variables))

    """
    Adding constraints
    """
    b = [C, np.sum(B), C1]  # Right-hand side of the inequalities
    for i in range(num_intervals):
        A_[0, num_intervals + i] = 1 # Charge to sell from battery from previous day cannot exceed the battery charge from previous day
        A_[1, num_intervals*2 + i] = 1 # Charge to sell from battery from present day cannot exceed the battery charge from previous day
        A_[1, i] = 1 # Charge to sell from panel + Charge to sell from battery from present day cannot exceed total panel output
        A_[2, num_intervals + i] = 1 # Charge to sell from battery from present day cannot exceed the battery charge from previous day
        A_[2, num_intervals*2 + i] = 1 # Charge to sell from battery from present day cannot exceed the battery charge from previous day
        A_[i+3, i] = 1 # Charge to sell from panel cannot exceed total panel output
        b.append(B[i])
    # Bounds for each variable (None means no bound)
    x_bounds = [(0, None) for i in range(24*4*3)]

    # Solve the problem
    result = linprog(-c, A_ub=A_, b_ub=b, bounds=x_bounds, method='highs')
    
    return result