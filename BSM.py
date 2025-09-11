import numpy as np
from math import exp
import matplotlib.pyplot as plt

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def bsm(s0, r, q, sigma, t):
    z = np.random.normal(0, 1)
    return round(s0*exp((r-q-0.5*sigma**2)*t+z*sigma*t**0.5), 2)


def simulation(simulation_time=10, s0=156.7, r=0.0015, q= 0.0233,sigma=0.282, t=0.5, sp=150, T=1):
    
    st_list=[[''for _ in range(4)]for _ in range(simulation_time)]
    # [[starting price, choice CALL/PUT, current price, current value] * simulation_time]
    discount_factor = exp(-r * T)

    for i in range(simulation_time):
        st_list[i][0] = bsm(s0, r, q, sigma, t)

        st_list[i][1] = 'CALL' if st_list[i][0] >150 else 'PUT' 

        st_list[i][2] = bsm(st_list[i][0], r, q, sigma, t)

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


simulation_time=100000


ct_list, pt_list = simulation(simulation_time=simulation_time)
ct_list, pt_list = transpose(ct_list), transpose(pt_list)
# [[starting price], [choice CALL/PUT], [current price], [current value]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list[2], y=ct_list[3], c='#ea7e36')
plt.scatter(x=pt_list[2], y=pt_list[3], c='#fcc01e')
plt.xlabel('Stock Price at Maturity ($)')
plt.ylabel('Option Payoff ($)')
plt.title('The 2nd Six-month Stock Price vs. Payoff')
plt.savefig('figs/The 2nd Six-month Stock Price vs. Payoff.png')
plt.clf()


ct_list_sigma, pt_list_sigma = [], []
for i in range(10, 90):
    sigma = i/100
    ct_list, pt_list = simulation(simulation_time=simulation_time, sigma=sigma)
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
plt.savefig('figs/Sensitivity Analysis on Sigma.png')
plt.clf()


ct_list_sp, pt_list_sp = [], []
for i in range(50, 450):
    sp = i
    ct_list, pt_list = simulation(simulation_time=simulation_time, sp=sp)
    ct_option_price = option_value(ct_list)
    ct_list_sp.append([ct_option_price, sp])
    pt_option_price = option_value(pt_list)
    pt_list_sp.append([pt_option_price, sp])


ct_list_sp, pt_list_sp = transpose(ct_list_sp), transpose(pt_list_sp)
# [[option value], [sp]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list_sp[1], y=ct_list_sp[0], c='#ea7e36')
plt.scatter(x=pt_list_sp[1], y=pt_list_sp[0], c='#fcc01e')
plt.xlabel('Strike Price')
plt.ylabel('Option Value')
plt.title('Sensitivity Analysis on the Strike Price')
plt.savefig('figs/Sensitivity Analysis on the the Strike Price.png')
plt.clf()


ct_list_r, pt_list_r = [], []
for i in range(0, 80):
    r = i/1000
    ct_list, pt_list = simulation(simulation_time=simulation_time, r=r)
    ct_option_price = option_value(ct_list)
    ct_list_r.append([ct_option_price, r])
    pt_option_price = option_value(pt_list)
    pt_list_r.append([pt_option_price, r])

ct_list_r, pt_list_r = transpose(ct_list_r), transpose(pt_list_r)
# [[option value], [r]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list_r[1], y=ct_list_r[0], c='#ea7e36')
plt.scatter(x=pt_list_r[1], y=pt_list_r[0], c='#fcc01e')
plt.xlabel('Risk-free Interest Rate')
plt.ylabel('Option Value')
plt.title('Sensitivity Analysis on the Risk-free Rate')
plt.savefig('figs/Sensitivity Analysis on the Risk-free Rate.png')
plt.clf()


ct_list_q, pt_list_q = [], []
for i in range(0, 80):
    q = i/1000
    ct_list, pt_list = simulation(simulation_time=simulation_time, q=q)
    ct_option_price = option_value(ct_list)
    ct_list_q.append([ct_option_price, q])
    pt_option_price = option_value(pt_list)
    pt_list_q.append([pt_option_price, q])

ct_list_q, pt_list_q = transpose(ct_list_q), transpose(pt_list_q)
# [[option value], [q]]

plt.figure(figsize=(10, 6))
plt.scatter(x=ct_list_q[1], y=ct_list_q[0], c='#ea7e36')
plt.scatter(x=pt_list_q[1], y=pt_list_q[0], c='#fcc01e')
plt.xlabel('Dividend Rate')
plt.ylabel('Option Value')
plt.title('Sensitivity Analysis on the Dividend Rate')
plt.savefig('figs/Sensitivity Analysis on the Dividend Rate.png')
plt.clf()
