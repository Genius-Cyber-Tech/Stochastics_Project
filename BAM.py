# ==== REQUIREMENTS ===
# numpy
# matplotlib
import numpy as np
import matplotlib.pyplot as plt

def binomial_option_trees(S0, u, d, r, t, strikes, option_type="call"): 
    # risk neutral probability
    q = (1 + r - d) / (u - d)
    print(f"The risk-neutral probability is: {q}")

    # create stock price tree
    S = np.zeros((t + 1, t + 1))
     # stock tree: S[i,n] where i=#up moves, n=time
    for n in range(t + 1):
        for i in range(n + 1):
           S[i,n] = (u**i) * (d**(n-i)) * S0
            # example: S[0,4] indicates price going down 4 times
            # example: S[4,4] indicates price going up 4 times      
             

    opt = {}  # empty dictionary for option value tree
    for K in strikes:
        H = np.zeros((t + 1, t + 1))

        # payoff at maturity
        for i in range(t + 1):
            if option_type == "call":
                H[i, t] = max(S[i, t] - K, 0.0)
            elif option_type == "put":
                H[i, t] = max(K - S[i, t], 0.0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

        # backward induction
        for n in range(t - 1, -1, -1):
            for i in range(n + 1):
                H[i, n] = (1 / (1 + r)) * (q * H[i + 1, n + 1] + (1 - q) * H[i, n + 1])

        opt[K] = H

    return q, S, opt

# Plotting function
def plot_evolution_root(opt_trees, strikes, t, title, ylabel):
    times = np.arange(t + 1)
    plt.figure()
    for K in strikes:
        series = [opt_trees[K][0, n] for n in times]
        plt.plot(times, series, marker="o", label=f"K={K}")
    plt.xlabel("time step n")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(times)
    plt.legend()
    plt.grid(True)
    plt.show()


# Input
S0, u, d, r, t = 100, 1.1, 0.9, 0.02, 4
strikes = [90, 95, 100, 105, 110]

q, S, call_trees = binomial_option_trees(S0, u, d, r, t, strikes, "call")
_, _, put_trees  = binomial_option_trees(S0, u, d, r, t, strikes, "put")

print("risk-neutral q =", q)
print("Stock tree S (rows i=#up, cols n=time):\n", S)


# Plots (as required)
plot_evolution_root(call_trees, strikes, t,
                    title="Evolution of Call Option Price over Time",
                    ylabel="Call price H[0,n]")

plot_evolution_root(put_trees, strikes, t,
                    title="Evolution of Put Option Price over Time",
                    ylabel="Put price H[0,n]")



