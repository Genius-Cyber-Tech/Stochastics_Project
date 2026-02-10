# ==== REQUIREMENTS ===
# numpy
# matplotlib
import numpy as np
import matplotlib.pyplot as plt

def price_tree(u, d, r, t, k, s): 
    # risk neutral probability
    q = (1 + r - d) / (u - d)
    print(f"The risk-neutral probability is: {q}")

    # create price tree
    tree = np.zeros((t + 1, t + 1))
    option_tree = np.zeros((t + 1, t + 1))
    # columns n = time steps
    for n in range(t + 1):
        for i in range(t + 1):
            s_n = (u**i) * (d**(n-i)) * s
            strike = k[0]
            # example: tree[0,4] indicates price going down 4 times
            # example: tree[4,4] indicates price going up 4 times
            tree[i, n] = s_n          

    # calculate options at last step
    for i in range(i + 1):
            #call option at prior steps for first strike (for now)
            #F_call(S_4) = max(S_4 - K, 0)
            option_tree[i, t] = max(tree[i, t] - strike, 0)

    for n in range(t - 1, -1, -1):
         #TO DO: calculate H_n at each step backwards
         # discount the stock price process 
        print("pending")

    print("Stock price grid: \n", tree)

    return q

# Input
up = 1.1
down = 0.9
rate = 0.02
time = 4
k_step = np.array([90, 95, 100, 105, 110]) # strikes
s_0 = float(input("Enter the beginning stock price: ")) #100

# Graphing
value = price_tree(up,down,rate,time,k_step,s_0)

# Messing around
plt.plot(5, 5) #will need to adjust later
plt.xlabel("time")
plt.ylabel("upward steps")

# test


