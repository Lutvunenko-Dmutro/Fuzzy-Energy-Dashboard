import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Початкова (стара) версія для порівняння (6 правил) ---
def get_initial_system_state(consumption_value, freq_dev_value):
    consumption = ctrl.Antecedent(np.arange(0, 101, 1), 'consumption')
    frequency_deviation = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'frequency_deviation')
    system_state = ctrl.Consequent(np.arange(0, 11, 1), 'system_state')

    consumption['Low'] = fuzz.trimf(consumption.universe, [0, 0, 40]); consumption['Normal'] = fuzz.trimf(consumption.universe, [30, 50, 70]); consumption['High'] = fuzz.trimf(consumption.universe, [60, 75, 90]); consumption['Critical'] = fuzz.trimf(consumption.universe, [85, 100, 100])
    frequency_deviation['Negative_High'] = fuzz.trimf(frequency_deviation.universe, [-0.5, -0.5, -0.2]); frequency_deviation['Negative_Low'] = fuzz.trimf(frequency_deviation.universe, [-0.3, -0.15, 0]); frequency_deviation['Stable'] = fuzz.trimf(frequency_deviation.universe, [-0.1, 0, 0.1]); frequency_deviation['Positive'] = fuzz.trimf(frequency_deviation.universe, [0.05, 0.5, 0.5])
    system_state['Stable'] = fuzz.trimf(system_state.universe, [0, 1.5, 3]); system_state['Warning'] = fuzz.trimf(system_state.universe, [2, 4.5, 7]); system_state['Danger'] = fuzz.trimf(system_state.universe, [6, 8, 10])

    rules = [ ctrl.Rule(consumption['Normal'] & frequency_deviation['Stable'], system_state['Stable']), ctrl.Rule(consumption['High'] & frequency_deviation['Negative_Low'], system_state['Warning']), ctrl.Rule(consumption['Critical'] & frequency_deviation['Negative_High'], system_state['Danger']), ctrl.Rule(consumption['Low'] & frequency_deviation['Positive'], system_state['Warning']), ctrl.Rule(consumption['High'] | consumption['Critical'], system_state['Danger']), ctrl.Rule(consumption['Low'], system_state['Stable']) ]
    
    power_grid_ctrl = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(power_grid_ctrl)
    simulation.input['consumption'] = consumption_value; simulation.input['frequency_deviation'] = freq_dev_value
    try:
        simulation.compute(); return simulation.output['system_state']
    except (KeyError, ValueError): return 0.0

# --- Оптимізована (нова) версія системи (20 правил) ---
def get_optimized_system_state(consumption_value, freq_dev_value):
    consumption = ctrl.Antecedent(np.arange(0, 101, 1), 'consumption')
    frequency_deviation = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'frequency_deviation')
    system_state = ctrl.Consequent(np.arange(0, 11, 1), 'system_state')

    # Оновлені функції належності
    consumption['Low'] = fuzz.trimf(consumption.universe, [0, 0, 30]); consumption['Normal'] = fuzz.trimf(consumption.universe, [25, 50, 75]); consumption['High'] = fuzz.trimf(consumption.universe, [70, 80, 90]); consumption['Critical'] = fuzz.trimf(consumption.universe, [85, 100, 100])
    frequency_deviation['Negative_High'] = fuzz.trimf(frequency_deviation.universe, [-0.5, -0.5, -0.2]); frequency_deviation['Negative_Low'] = fuzz.trimf(frequency_deviation.universe, [-0.25, -0.1, 0]); frequency_deviation['Stable'] = fuzz.trimf(frequency_deviation.universe, [-0.05, 0, 0.05]); frequency_deviation['Positive'] = fuzz.trimf(frequency_deviation.universe, [0.05, 0.5, 0.5])
    system_state['Stable'] = fuzz.trimf(system_state.universe, [0, 1.5, 3]); system_state['Warning'] = fuzz.trimf(system_state.universe, [2.5, 4.5, 7]); system_state['Danger'] = fuzz.trimf(system_state.universe, [6.5, 8, 10])
    # Старі правила
    rules = [
        # --- Стабільні сценарії ---
        ctrl.Rule(consumption['Low'] & frequency_deviation['Stable'], system_state['Stable']),
        ctrl.Rule(consumption['Low'] & frequency_deviation['Negative_Low'], system_state['Stable']),
        ctrl.Rule(consumption['Low'] & frequency_deviation['Positive'], system_state['Stable']),
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Stable'], system_state['Stable']),
        # --- Сценарії попередження ---
        ctrl.Rule(consumption['Low'] & frequency_deviation['Negative_High'], system_state['Warning']),
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Negative_Low'], system_state['Warning']),
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Positive'], system_state['Warning']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Stable'], system_state['Warning']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Negative_Low'], system_state['Warning']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Positive'], system_state['Warning']),
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Stable'], system_state['Warning']),
        # --- Небезпечні сценарії ---
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Positive'], system_state['Danger']),
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Negative_High'], system_state['Danger']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Negative_High'], system_state['Danger']),
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Negative_Low'], system_state['Danger']),
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Negative_High'], system_state['Danger']),
        # --- Загальні правила безпеки (спрацьовують, якщо нічого іншого не підійшло) ---
        ctrl.Rule(consumption['Critical'], system_state['Danger']),
        ctrl.Rule(frequency_deviation['Negative_High'], system_state['Danger']),
        ctrl.Rule(consumption['High'], system_state['Warning']),
        ctrl.Rule(consumption['Low'], system_state['Stable'])
    ]
    
    power_grid_ctrl = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(power_grid_ctrl)
    
    simulation.input['consumption'] = consumption_value
    simulation.input['frequency_deviation'] = freq_dev_value
    
    try:
        simulation.compute()
        return simulation.output['system_state']
    except (KeyError, ValueError):
        return 0.0

def get_optimized_system_state(consumption_value, freq_dev_value):
    consumption = ctrl.Antecedent(np.arange(0, 101, 1), 'consumption')
    frequency_deviation = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'frequency_deviation')
    system_state = ctrl.Consequent(np.arange(0, 11, 1), 'system_state')

    consumption['Low'] = fuzz.trimf(consumption.universe, [0, 0, 30])
    consumption['Normal'] = fuzz.trimf(consumption.universe, [25, 50, 75]) 
    consumption['High'] = fuzz.trimf(consumption.universe, [70, 80, 90])
    consumption['Critical'] = fuzz.trimf(consumption.universe, [85, 100, 100])
    
    frequency_deviation['Negative_High'] = fuzz.trimf(frequency_deviation.universe, [-0.5, -0.5, -0.2])
    frequency_deviation['Negative_Low'] = fuzz.trimf(frequency_deviation.universe, [-0.25, -0.1, 0])
    frequency_deviation['Stable'] = fuzz.trimf(frequency_deviation.universe, [-0.05, 0, 0.05]) 
    frequency_deviation['Positive'] = fuzz.trimf(frequency_deviation.universe, [0.05, 0.5, 0.5])

    system_state['Stable'] = fuzz.trimf(system_state.universe, [0, 1.5, 3])
    system_state['Warning'] = fuzz.trimf(system_state.universe, [2.5, 4.5, 7])
    system_state['Danger'] = fuzz.trimf(system_state.universe, [6.5, 8, 10])

    rules = [
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Stable'], system_state['Stable']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Negative_Low'], system_state['Warning']),
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Negative_High'], system_state['Danger']),
        ctrl.Rule(consumption['Low'] & frequency_deviation['Positive'], system_state['Warning']),
        ctrl.Rule(consumption['High'] | consumption['Critical'], system_state['Danger']),
        ctrl.Rule(consumption['Low'], system_state['Stable']),
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Negative_Low'], system_state['Warning']), 
        ctrl.Rule(consumption['High'] & frequency_deviation['Stable'], system_state['Warning']) 
    ]
    
    power_grid_ctrl = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(power_grid_ctrl)
    
    simulation.input['consumption'] = consumption_value
    simulation.input['frequency_deviation'] = freq_dev_value
    
    try:
        simulation.compute()
        return simulation.output['system_state']
    except (KeyError, ValueError):
        return 0.0