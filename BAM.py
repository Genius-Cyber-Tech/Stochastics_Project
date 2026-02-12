# ==== REQUIREMENTS ===
# numpy
# matplotlib
import numpy as np
import matplotlib.pyplot as plt

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
            'H': H
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


#Calculate the portfolios corresponding to the call and put option via the above defined function
call_portfolios = compute_replicating_portfolios(S, call_trees, u, d, r, t, strikes)
put_portfolios = compute_replicating_portfolios(S, put_trees, u, d, r, t, strikes)

# Plot portfolio values and option prices next to one another for comparison
plot_portfolio_and_option(call_portfolios, strikes, t, "call")
plot_portfolio_and_option(put_portfolios, strikes, t, "put")



