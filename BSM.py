import numpy as np
from math import exp
import matplotlib.pyplot as plt

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def bsm(s0, r, sigma, t):
    z = np.random.normal(0, 1)
    return round(s0*exp((r-0.5*sigma**2)*t+z*sigma*t**0.5), 2)


def simulation(simulation_time=10, s0=156.7, r=0.0015, sigma=0.282, t=0.5, sp=150, T=1):
    
    st_list=[[''for _ in range(4)]for _ in range(simulation_time)]
    # [[starting price, choice CALL/PUT, current price, current value] * simulation_time]
    discount_factor = exp(-r * T)

    for i in range(simulation_time):
        st_list[i][0] = bsm(s0, r, sigma, t)

        st_list[i][1] = 'CALL' if st_list[i][0] >150 else 'PUT' 

        st_list[i][2] = bsm(st_list[i][0], r, sigma, t)

        future_payoff = max((st_list[i][2] - sp if st_list[i][1] == 'CALL' else sp - st_list[i][2]), 0)
        st_list[i][3] = round(future_payoff * discount_factor, 2)

    ct_list, pt_list = [], []

    for row in st_list:
        if row[1] == 'CALL':
            ct_list.append(row)
        else:
            pt_list.append(row)

    return (ct_list, pt_list)


def option_value(st_list, r=0.0015, T=1.0):
    count, sum = 0, 0
    for row in st_list:
        sum += row[3]
        count += 1
    try:
        average_current_value = sum/count
    except:
        average_current_value = 0
    return average_current_value


ct_list, pt_list = simulation(simulation_time=10000)
ct_list, pt_list = transpose(ct_list), transpose(pt_list)
# [[starting price], [choice CALL/PUT], [current price], [current value]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list[2], y=ct_list[3], c='#ea7e36')
plt.scatter(x=pt_list[2], y=pt_list[3], c='#fcc01e')
plt.xlabel('Stock Price at Maturity ($)')
plt.ylabel('Option Payoff ($)')
plt.title('The 2nd Six-month Stock Price vs. Payoff')
plt.show()
plt.clf()


ct_list_sigma, pt_list_sigma = [], []
np.random.seed(466)
for i in range(10, 90):
    sigma = i/100
    ct_list, pt_list = simulation(simulation_time=10000, sigma=sigma)
    ct_option_price = option_value(ct_list)
    ct_list_sigma.append([ct_option_price, sigma])
    pt_option_price = option_value(pt_list)
    pt_list_sigma.append([pt_option_price, sigma])


ct_list_sigma, pt_list_sigma = transpose(ct_list_sigma), transpose(pt_list_sigma)
# [[option value], [sigma]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list_sigma[1], y=ct_list_sigma[0], c='#ea7e36')
plt.scatter(x=pt_list_sigma[1], y=pt_list_sigma[0], c='#fcc01e')
plt.xlabel('Volatility (sigma)')
plt.ylabel('Option Value')
plt.title('Sensitivity Analysis on Sigma')
plt.show()
plt.clf()
