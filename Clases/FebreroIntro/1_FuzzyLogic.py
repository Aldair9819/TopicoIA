#Python 3.8.18
import time
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)

# You can see how these look with .view()
quality["good"].view()
service.view()



# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])
tip['high'].view()

quality.view()
service.view()
tip.view()

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])
rule4 = ctrl.Rule(service['poor'] | quality['good'], tip['medium'])
rule5 = ctrl.Rule(service['average'] | quality['good'], tip['high'])

rule1.view()

tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
tipping_ctrl.view()

# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
tipping.input['quality'] = 7
tipping.input['service'] = 5

# Crunch the numbers
tipping.compute()
print(tipping.output['tip'])
service.view(sim= tipping)
quality.view(sim= tipping)
tip.view(sim=tipping)

plt.savefig("quality_membership.png")
plt.savefig("service_membership.png")

time.sleep(10)