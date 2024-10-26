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


import numpy as np

def dynamic_charge_discharge(num_intervals, market_price, plant_production, initial_battery_charge, battery_capacity):
    direct_sale = np.zeros(num_intervals)
    battery_sale = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    
    for t in range(num_intervals):
        if market_price[t] > np.mean(market_price):  # Discharge when price is high
            sell_from_battery = battery_charge
            battery_sale[t] = sell_from_battery
            battery_charge -= sell_from_battery
        else:  # Charge when price is low
            charge_amount = min(plant_production[t], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sale[t] = plant_production[t] - charge_amount

    return direct_sale, battery_sale

def price_threshold_strategy(num_intervals, market_price, plant_production, initial_battery_charge, battery_capacity):
    threshold = np.mean(market_price)
    direct_sale = np.zeros(num_intervals)
    battery_sale = np.zeros(num_intervals)
    battery_charge = initial_battery_charge

    for t in range(num_intervals):
        if market_price[t] > threshold:  # Discharge if price above threshold
            sell_from_battery = battery_charge
            battery_sale[t] = sell_from_battery
            battery_charge -= sell_from_battery
        else:  # Charge if price below threshold
            charge_amount = min(plant_production[t], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sale[t] = plant_production[t] - charge_amount

    return direct_sale, battery_sale

def peak_shaving_strategy(num_intervals, market_price, plant_production, initial_battery_charge, battery_capacity):
    high_price_threshold = np.quantile(market_price, 0.75)
    direct_sale = np.zeros(num_intervals)
    battery_sale = np.zeros(num_intervals)
    battery_charge = initial_battery_charge

    for t in range(num_intervals):
        if market_price[t] >= high_price_threshold:  # Only discharge during high price peaks
            sell_from_battery = battery_charge
            battery_sale[t] = sell_from_battery
            battery_charge -= sell_from_battery
        else:  # Otherwise, charge the battery
            charge_amount = min(plant_production[t], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sale[t] = plant_production[t] - charge_amount

    return direct_sale, battery_sale

def forecast_adaptive_strategy(num_intervals, market_price, plant_production, initial_battery_charge, battery_capacity):
    direct_sale = np.zeros(num_intervals)
    battery_sale = np.zeros(num_intervals)
    battery_charge = initial_battery_charge

    for t in range(num_intervals):
        if market_price[t] > np.mean(market_price[:t+1]):  # Adaptive threshold based on past prices
            sell_from_battery = battery_charge
            battery_sale[t] = sell_from_battery
            battery_charge -= sell_from_battery
        else:
            charge_amount = min(plant_production[t], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sale[t] = plant_production[t] - charge_amount

    return direct_sale, battery_sale

def scenario_based_strategy(num_intervals, market_price, plant_production, initial_battery_charge, battery_capacity):
    scenarios = {'low': 0.5 * plant_production, 'medium': plant_production, 'high': 1.5 * plant_production}
    best_scenario = max(scenarios, key=lambda scenario: np.dot(scenarios[scenario], market_price))
    
    direct_sale = np.zeros(num_intervals)
    battery_sale = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    selected_production = scenarios[best_scenario]

    for t in range(num_intervals):
        if market_price[t] > np.mean(market_price):  # Discharge in high-price windows
            sell_from_battery = battery_charge
            battery_sale[t] = sell_from_battery
            battery_charge -= sell_from_battery
        else:  # Charge in low-price windows
            charge_amount = min(selected_production[t], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sale[t] = selected_production[t] - charge_amount

    return direct_sale, battery_sale


import numpy as np

def dynamic_scheduling(num_intervals, prices, production, initial_battery_charge, battery_capacity):
    # Dynamic scheduling approach
    direct_sales = np.zeros(num_intervals)
    battery_sales = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    
    for i in range(num_intervals):
        if battery_charge < battery_capacity and production[i] > 0:
            # Charge battery with available production
            charge_amount = min(production[i], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sales[i] = production[i] - charge_amount
        else:
            direct_sales[i] = production[i]
        
        # Discharge battery if price is high enough
        if prices[i] > np.mean(prices) and battery_charge > 0:
            battery_sales[i] = min(battery_charge, production[i])
            battery_charge -= battery_sales[i]
    
    return direct_sales, battery_sales

def price_threshold(num_intervals, prices, production, initial_battery_charge, battery_capacity):
    # Price threshold approach
    direct_sales = np.zeros(num_intervals)
    battery_sales = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    threshold = np.mean(prices)
    
    for i in range(num_intervals):
        if prices[i] < threshold and battery_charge < battery_capacity:
            # Charge battery
            charge_amount = min(production[i], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sales[i] = production[i] - charge_amount
        elif prices[i] >= threshold and battery_charge > 0:
            # Discharge battery
            battery_sales[i] = min(battery_charge, production[i])
            battery_charge -= battery_sales[i]
        else:
            direct_sales[i] = production[i]
    
    return direct_sales, battery_sales

def peak_shaving(num_intervals, prices, production, initial_battery_charge, battery_capacity):
    # Peak shaving approach
    direct_sales = np.zeros(num_intervals)
    battery_sales = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    high_price_threshold = np.percentile(prices, 75)  # Top 25% price as high price
    
    for i in range(num_intervals):
        if prices[i] >= high_price_threshold and battery_charge > 0:
            # Discharge battery during high price
            battery_sales[i] = min(battery_charge, production[i])
            battery_charge -= battery_sales[i]
        elif production[i] > 0 and battery_charge < battery_capacity:
            # Charge battery
            charge_amount = min(production[i], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sales[i] = production[i] - charge_amount
        else:
            direct_sales[i] = production[i]
    
    return direct_sales, battery_sales

def adaptive_algorithm(num_intervals, prices, production, initial_battery_charge, battery_capacity):
    # Forecast-driven adaptive algorithm
    direct_sales = np.zeros(num_intervals)
    battery_sales = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    
    for i in range(num_intervals):
        forecasted_high_price = np.max(prices[i:]) if i < num_intervals - 1 else prices[i]
        
        if prices[i] < forecasted_high_price and battery_charge < battery_capacity:
            # Charge battery if lower than expected high price
            charge_amount = min(production[i], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sales[i] = production[i] - charge_amount
        elif prices[i] >= forecasted_high_price and battery_charge > 0:
            # Discharge during forecasted peak
            battery_sales[i] = min(battery_charge, production[i])
            battery_charge -= battery_sales[i]
        else:
            direct_sales[i] = production[i]
    
    return direct_sales, battery_sales

def scenario_optimization(num_intervals, prices, production, initial_battery_charge, battery_capacity):
    # Scenario-based optimization (basic version)
    direct_sales = np.zeros(num_intervals)
    battery_sales = np.zeros(num_intervals)
    battery_charge = initial_battery_charge
    price_scenarios = [np.percentile(prices, 25), np.median(prices), np.percentile(prices, 75)]
    
    for i in range(num_intervals):
        current_scenario = price_scenarios[min(i // (num_intervals // 3), 2)]
        
        if prices[i] < current_scenario and battery_charge < battery_capacity:
            # Charge battery in low scenario
            charge_amount = min(production[i], battery_capacity - battery_charge)
            battery_charge += charge_amount
            direct_sales[i] = production[i] - charge_amount
        elif prices[i] >= current_scenario and battery_charge > 0:
            # Discharge battery in high scenario
            battery_sales[i] = min(battery_charge, production[i])
            battery_charge -= battery_sales[i]
        else:
            direct_sales[i] = production[i]
    
    return direct_sales, battery_sales
