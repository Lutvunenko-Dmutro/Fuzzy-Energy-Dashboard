import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

_initial_ctrl = None
_optimized_ctrl = None

def _build_initial_ctrl():
    global _initial_ctrl
    if _initial_ctrl is not None:
        return
    consumption = ctrl.Antecedent(np.arange(0, 101, 1), 'consumption')
    frequency_deviation = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'frequency_deviation')
    system_state = ctrl.Consequent(np.arange(0, 11, 1), 'system_state')

    consumption['Low'] = fuzz.trimf(consumption.universe, [0, 0, 40])
    consumption['Normal'] = fuzz.trimf(consumption.universe, [30, 50, 70])
    consumption['High'] = fuzz.trimf(consumption.universe, [60, 75, 90])
    consumption['Critical'] = fuzz.trimf(consumption.universe, [85, 100, 100])

    frequency_deviation['Negative_High'] = fuzz.trimf(frequency_deviation.universe, [-0.5, -0.5, -0.2])
    frequency_deviation['Negative_Low'] = fuzz.trimf(frequency_deviation.universe, [-0.3, -0.15, 0])
    frequency_deviation['Stable'] = fuzz.trimf(frequency_deviation.universe, [-0.1, 0, 0.1])
    frequency_deviation['Positive'] = fuzz.trimf(frequency_deviation.universe, [0.05, 0.5, 0.5])

    system_state['Stable'] = fuzz.trimf(system_state.universe, [0, 1.5, 3])
    system_state['Warning'] = fuzz.trimf(system_state.universe, [2, 4.5, 7])
    system_state['Danger'] = fuzz.trimf(system_state.universe, [6, 8, 10])

    rules = [
        ctrl.Rule(consumption['Normal'] & frequency_deviation['Stable'], system_state['Stable']),
        ctrl.Rule(consumption['High'] & frequency_deviation['Negative_Low'], system_state['Warning']),
        ctrl.Rule(consumption['Critical'] & frequency_deviation['Negative_High'], system_state['Danger']),
        ctrl.Rule(consumption['Low'] & frequency_deviation['Positive'], system_state['Warning']),
        ctrl.Rule(consumption['High'] | consumption['Critical'], system_state['Danger']),
        ctrl.Rule(consumption['Low'], system_state['Stable'])
    ]
    _initial_ctrl = ctrl.ControlSystem(rules)

def _build_optimized_ctrl():
    global _optimized_ctrl
    if _optimized_ctrl is not None:
        return
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
    
    multiplier_factor = 6
    cap = 12

    rules_specs = [
        # Повна матриця 4x4 (consumption × frequency)
        (consumption['Low']     & frequency_deviation['Stable'],       system_state['Stable'], 0.3),
        (consumption['Low']     & frequency_deviation['Negative_Low'], system_state['Stable'], 0.35),
        (consumption['Low']     & frequency_deviation['Negative_High'],system_state['Warning'],0.6),
        (consumption['Low']     & frequency_deviation['Positive'],     system_state['Stable'], 0.3),

        (consumption['Normal']  & frequency_deviation['Stable'],       system_state['Stable'], 0.5),
        (consumption['Normal']  & frequency_deviation['Negative_Low'], system_state['Warning'],0.75),
        (consumption['Normal']  & frequency_deviation['Negative_High'],system_state['Danger'], 1.0),
        (consumption['Normal']  & frequency_deviation['Positive'],     system_state['Warning'],0.7),

        (consumption['High']    & frequency_deviation['Stable'],       system_state['Warning'],0.85),
        (consumption['High']    & frequency_deviation['Negative_Low'], system_state['Warning'],1.0),
        (consumption['High']    & frequency_deviation['Negative_High'],system_state['Danger'], 1.15),
        (consumption['High']    & frequency_deviation['Positive'],     system_state['Warning'],0.95),

        (consumption['Critical']& frequency_deviation['Stable'],       system_state['Warning'],0.9),
        (consumption['Critical']& frequency_deviation['Negative_Low'], system_state['Danger'], 1.2),
        (consumption['Critical']& frequency_deviation['Negative_High'],system_state['Danger'], 1.3),
        (consumption['Critical']& frequency_deviation['Positive'],     system_state['Danger'], 1.25),

        # Комбіновані правила між рівнями споживання
        ((consumption['Low'] | consumption['Normal'])   & frequency_deviation['Stable'], system_state['Stable'], 0.4),
        ((consumption['Normal'] | consumption['High'])  & frequency_deviation['Negative_Low'], system_state['Warning'], 0.85),
        ((consumption['High'] | consumption['Critical'])& frequency_deviation['Positive'], system_state['Danger'], 1.15),
        ((consumption['Low'] | consumption['Normal'])   & frequency_deviation['Negative_High'], system_state['Warning'], 0.8),

        # Уточнюючі та переходні правила (щоби краще покривати границі)
        (consumption['Low']     & (frequency_deviation['Stable'] | frequency_deviation['Negative_Low']), system_state['Stable'], 0.35),
        (consumption['Normal']  & (frequency_deviation['Positive'] | frequency_deviation['Negative_Low']), system_state['Warning'], 0.7),
        (consumption['High']    & (frequency_deviation['Stable'] | frequency_deviation['Negative_Low']), system_state['Warning'], 0.9),
        (consumption['Critical']& (frequency_deviation['Stable'] | frequency_deviation['Negative_Low']), system_state['Warning'], 0.95),

        # Додаткові "захисні" правила, щоб система була консервативнішою при екстремумах
        (consumption['Critical'],                            system_state['Danger'], 0.5),
        (frequency_deviation['Negative_High'],               system_state['Danger'], 0.7),
        ((consumption['High'] & frequency_deviation['Negative_High']) | (consumption['Critical'] & frequency_deviation['Negative_High']), system_state['Danger'], 1.2),
        ((consumption['High'] & frequency_deviation['Positive']) & (~consumption['Low']), system_state['Warning'], 0.9),

        # Розширені специфічні кейси (комбінації трьох умов через логіку OR/AND)
        ((consumption['Normal'] | consumption['High']) & (frequency_deviation['Negative_High'] | frequency_deviation['Negative_Low']), system_state['Danger'], 1.05),
        ((consumption['Low'] | consumption['Normal']) & (frequency_deviation['Stable'] | frequency_deviation['Positive']), system_state['Stable'], 0.4),
        ((consumption['High'] | consumption['Critical']) & (frequency_deviation['Negative_High'] | frequency_deviation['Positive']), system_state['Danger'], 1.15),

        # Просте fallback / low-weight rules
        (consumption['High'],  system_state['Warning'], 0.35),
        (consumption['Low'],   system_state['Stable'],  0.25),
    ]
    rules = []
    for cond, consq, w in rules_specs:
        repeats = int(round(w * multiplier_factor))
        repeats = max(1, min(repeats, cap))
        for _ in range(repeats):
            rules.append(ctrl.Rule(cond, consq))

    _optimized_ctrl = ctrl.ControlSystem(rules)

def get_initial_system_state(consumption_value, freq_dev_value):
    _build_initial_ctrl()
    simulation = ctrl.ControlSystemSimulation(_initial_ctrl)
    simulation.input['consumption'] = consumption_value
    simulation.input['frequency_deviation'] = freq_dev_value
    try:
        simulation.compute()
        return float(simulation.output['system_state'])
    except (KeyError, ValueError):
        return 0.0

def get_optimized_system_state(consumption_value, freq_dev_value):
    _build_optimized_ctrl()
    simulation = ctrl.ControlSystemSimulation(_optimized_ctrl)
    simulation.input['consumption'] = consumption_value
    simulation.input['frequency_deviation'] = freq_dev_value
    try:
        simulation.compute()
        return float(simulation.output['system_state'])
    except (KeyError, ValueError):
        return 0.0
