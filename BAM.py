# ==== REQUIREMENTS ===
# numpy
# matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt, erf

# ==== PART A ===
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
        # want nodes that were actually calculated to take the avg
        series = []
        for n in times:
            # get active nodes at time n
            nodes = opt_trees[K][:n+1, n]
            series.append(np.mean(nodes))
        plt.plot(times, series, marker="o", label=f"K={K}")
    plt.xlabel("time step n")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(times)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_partA_all_nodes(opt_trees, strikes, t):
    plt.figure(figsize=(8, 5))

    for K in strikes:
        H = opt_trees[K]
        for n in range(t + 1):
            x = np.full(n + 1, n)     # time n
            y = H[:n + 1, n]          # all nodes at time n
            plt.scatter(
                x, y,
                alpha=0.7,
                label=f"K={K}" if n == 0 else ""
            )
    plt.xlabel("time step n")
    plt.ylabel("option price")
    plt.title("Part A: Option prices at all nodes (binomial model)")
    plt.xticks(range(t + 1))
    plt.grid(True)
    plt.legend()
    plt.show()

# Input
S0, u, d, r, t = 100, 1.1, 0.9, 0.02, 4
strikes = [90, 95, 100, 105, 110]

q, S, call_trees = binomial_option_trees(S0, u, d, r, t, strikes, "call")
_, _, put_trees  = binomial_option_trees(S0, u, d, r, t, strikes, "put")

print("risk-neutral q =", q)
print("Stock tree S (rows i=#up, cols n=time):\n", S)


# Plots (as required)
# Plot with the average of the nodes at each time step to get a single line per strike price, for better visualization of the evolution of the option price over time.
plot_evolution_root(call_trees, strikes, t,
                    title="Evolution of Call Option Price over Time",
                    ylabel="Call price H[0,n]")

plot_evolution_root(put_trees, strikes, t,
                    title="Evolution of Put Option Price over Time",
                    ylabel="Put price H[0,n]")

#Plot all nodes at each time 
plot_partA_all_nodes(call_trees, strikes, t)
plot_partA_all_nodes(put_trees, strikes, t)

# ==== PART B ===

#Define a function that computes a dictionary containing the different portfolios depending on the strike prices 

def compute_replicating_portfolios(S, opt_trees, u, d, r, t, strikes):
    
    """
    The goal is to compute the values of the replicating portfolio at each time step, dependent on the 
    amount of up-movements of the risky asset S, via solving a linear system of equations given in the script and below.
    """

    #empty dictionary
    portfolios = {}

    for K in strikes:
        
        alpha = np.zeros((t + 1, t + 1))
        beta = np.zeros((t + 1, t + 1))
        V = np.zeros((t + 1, t + 1))
        H = opt_trees[K]
        
        # At the temporal endpoint, the option payoff equals the value of the portfolio:
        V[:, t] = H[:, t]
    
        # Backward induction 
        for n in range(t - 1, -1, -1):
            B_n = (1 + r) ** n
            B_n1 = (1 + r) ** (n + 1)
        
            for i in range(n + 1):
                S_n = S[i, n]
            
                # Values in up and down states at time n+1
                V_up = V[i + 1, n + 1]
                V_down = V[i, n + 1]
            
                # Stock prices in up and down states
                S_up = u * S_n
                S_down = d * S_n
                
                # Solve the following linear system analytically:
                # alpha * S_up + beta * B_n1 = V_up
                # alpha * S_down + beta * B_n1 = V_down
            
                # Via subtracting equations from one another:
                alpha[i, n] = (V_up - V_down) / (S_up - S_down)
                
                # Substitution to find beta:
                beta[i, n] = (V_up - alpha[i, n] * S_up) / B_n1
            
                # Calculate portfolio value, which should match H[i,n]
                V[i, n] = alpha[i, n] * S_n + beta[i, n] * B_n
    
        """
        A dictionary is used to compute the replicating portfolios corresponding to different strike prizes.
        """
        portfolios[K] = {
            'V': V,
            'H': H,
            'alpha': alpha,
            'beta': beta
        }   
    
    return portfolios

#Define a function for plotting the portfolio values as well as the option princes and compare them

def plot_portfolio_and_option(portfolios, strikes, t, option_type):
    """
    Plot the portfolio value V[0,n] compared to the option price H[0,n] (root path).
    Equality is expected.
    """
    times = np.arange(t + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot portfolio values
    for K in strikes:
        V_root = [portfolios[K]['V'][0, n] for n in times]
        ax1.plot(times, V_root, marker='o', label=f'K={K}')
    
    ax1.set_xlabel('Time step n')
    ax1.set_ylabel('Portfolio value V[0,n]')
    ax1.set_title(f'Portfolio Value Replicating the {option_type.capitalize()}-Option over Time')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(times)
    
    # Plot option prices
    for K in strikes:
        H_root = [portfolios[K]['H'][0, n] for n in times]
        ax2.plot(times, H_root, marker='s', label=f'K={K}')
    
    ax2.set_xlabel('Time step n')
    ax2.set_ylabel('Option price H[0,n]')
    ax2.set_title(f'Option Price of the {option_type.capitalize()}-Option over Time')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(times)
    
    plt.show()
    
    # To check the difference between portfolio value and the option prize, the maximal difference is taken 
    # to ensure replication of the portfolio: 
    for K in strikes:
        max_diff = np.max(np.abs(portfolios[K]['V'] - portfolios[K]['H']))
        print(f" For K={K}: Maximal difference between V and H = {max_diff:.1e}")


def print_alpha_beta(portfolios, strikes, t):
    """
    Print alpha_n and beta_n at each node (n,i) for each strike.
    """
    for K in strikes:
        print(f"\n========== Replicating portfolio for K = {K} ==========")
        alpha = portfolios[K]['alpha']
        beta = portfolios[K]['beta']
        
        for n in range(t):
            print(f"\nTime step n = {n}:")
            for i in range(n + 1):
                print(
                    f"Node (i={i}): "
                    f"alpha = {alpha[i, n]: .6f}, "
                    f"beta = {beta[i, n]: .6f}"
                )

#Calculate the portfolios corresponding to the call and put option via the above defined function
call_portfolios = compute_replicating_portfolios(S, call_trees, u, d, r, t, strikes)
put_portfolios = compute_replicating_portfolios(S, put_trees, u, d, r, t, strikes)

# Plot portfolio values and option prices next to one another for comparison
plot_portfolio_and_option(call_portfolios, strikes, t, "call")
plot_portfolio_and_option(put_portfolios, strikes, t, "put")

# Print alpha and beta values at each node for each strike price
print_alpha_beta(call_portfolios, strikes, t)
print_alpha_beta(put_portfolios, strikes, t)



# ==== PART C ===

# ---- Normal CDF without scipy ----
def norm_cdf(x: float):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call_price(S0, K, r, sigma, T):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def binomial_call_price_CRR(S0, K, r, sigma, T, N):
    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = exp(-sigma * sqrt(dt))
    r_step = exp(r * dt) - 1  # convert cont. rate to per-step simple rate

    _, _, opt = binomial_option_trees(S0, u, d, r_step, N, [K], option_type="call")
    return opt[K][0, 0]

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
Ns = [5, 10, 20, 50, 100, 1000]

bs_price = bs_call_price(S0, K, r, sigma, T)
bin_prices = [binomial_call_price_CRR(S0, K, r, sigma, T, N) for N in Ns]
errors = [abs(p - bs_price) for p in bin_prices]

print("Black–Scholes price:", bs_price)
for N, p, e in zip(Ns, bin_prices, errors):
    print(f"N={N:4d}  Binomial={p:.6f}  Error={e:.6e}")

plt.figure()
plt.plot(Ns, bin_prices, marker='o', label="Binomial (CRR)")
plt.axhline(bs_price, linestyle='--', label="Black–Scholes")
plt.xscale('log')
plt.xlabel("N (log scale)")
plt.ylabel("Call price at t=0")
plt.title("Binomial (CRR) price convergence to Black–Scholes")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(Ns, errors, marker='o')
plt.xscale('log'); plt.yscale('log')
plt.xlabel("N (log scale)")
plt.ylabel("|Binomial - Black–Scholes| (log scale)")
plt.title("Convergence error")
plt.grid(True)
plt.show()

