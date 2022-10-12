import matplotlib.pyplot as plt
import numpy as np


years = np.linspace(0,1,4)

wage = 27*8*365
employee = wage*years


power = .300*0.296*8*365
print(power)
maintenance = 400
ur5 = 40000
tooling = 5000
programming = 40 * 200
robot = (maintenance+power)*years+(ur5+tooling+programming)

coffee_margin = 4.80 * 0.8
coffee_per_day = 230
revenue_per_year = coffee_margin*coffee_per_day*365
revenue = revenue_per_year*years

plt.plot(years, employee/1000, label='Employee costs')
plt.plot(years, robot/1000, label='UR5 costs')
plt.plot(years, revenue/1000, label='Sales revenue')

plt.xlabel("Years")
plt.ylabel("$1000's")
plt.legend(loc='upper left')
plt.grid()

plt.show()