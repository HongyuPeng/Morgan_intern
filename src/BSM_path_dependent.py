import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def generate_paths(s0, r, q, sigma, T, n_steps, n_paths):
    """
    Vectorized GBM stock price paths
    """
    dt = T / n_steps
    z = np.random.normal(0, 1, (n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])  # include S0
    paths = s0 * np.exp(log_paths)
    return paths, dt


def asian_option_value(paths, K, r, T, option_type='call', start_step=0):
    """
    Vectorized Asian option value from start_step to end
    """
    averages = np.mean(paths[:, start_step:], axis=1)
    if option_type.lower() == 'call':
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)

    remaining_time = T * (paths.shape[1] - 1 - start_step) / (paths.shape[1] - 1)
    values = np.exp(-r * remaining_time) * payoffs
    return values


def lookback_option_value(paths, K, r, T, option_type='call', start_step=0):
    """
    Vectorized Lookback option value from start_step
    """
    if option_type.lower() == 'call':
        max_prices = np.max(paths[:, start_step:], axis=1)
        payoffs = np.maximum(max_prices - K, 0)
    else:
        min_prices = np.min(paths[:, start_step:], axis=1)
        payoffs = np.maximum(K - min_prices, 0)

    remaining_time = T * (paths.shape[1] - 1 - start_step) / (paths.shape[1] - 1)
    values = np.exp(-r * remaining_time) * payoffs
    return values


def path_dependent_chooser_option(s0, K, r, q, sigma, T, choice_date, n_steps, n_paths, option_type='asian'):
    """
    Vectorized Path-dependent Chooser Option Monte Carlo pricing
    """
    # Generate paths
    paths, dt = generate_paths(s0, r, q, sigma, T, n_steps, n_paths)
    choice_step = int(choice_date / T * n_steps)

    # Compute call & put values (vectorized)
    if option_type == 'asian':
        call_values = asian_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = asian_option_value(paths, K, r, T, 'put', start_step=choice_step)
    elif option_type == 'lookback':
        call_values = lookback_option_value(paths, K, r, T, 'call', start_step=choice_step)
        put_values = lookback_option_value(paths, K, r, T, 'put', start_step=choice_step)
    else:
        raise ValueError("option_type must be 'asian' or 'lookback'")

    # Choose the better option type
    choices = np.where(call_values > put_values, 'CALL', 'PUT')

    # Final payoffs depend on choices
    final_payoffs = np.where(
        choices == 'CALL',
        call_values,
        put_values
    )

    # Discount back to today
    option_price = np.exp(-r * choice_date) * np.mean(final_payoffs)
    std_error = np.std(final_payoffs) / np.sqrt(n_paths)

    return option_price, std_error, choices, call_values, put_values, paths


if __name__ == '__main__':
    
    # =============================================================================
    # Example: Asian Chooser Option
    # =============================================================================

    # Set parameters
    params = {
        's0': 156.7,           # Initial stock price
        'K': 150,            # Strike price
        'r': 0.0015,           # Risk-free rate
        'q': 0.0233,           # Dividend yield
        'sigma': 0.282,        # Volatility
        'T': 1.0,            # Total time to expiration
        'choice_date': 0.5,  # Choice date (choose after half year)
        'n_steps': 252,      # Number of time steps
        'n_paths': 500000,    # Number of simulation paths
        'option_type': 'asian'  # Path-dependent type: 'asian' or 'lookback'
    }


    print("=" * 70)
    print("Path-Dependent Chooser Option Monte Carlo Simulation")
    print("=" * 70)

    # Run simulation
    price, std_error, choices, call_vals, put_vals, paths = path_dependent_chooser_option(**params)

    # Analyze results
    call_choice_ratio = np.sum(choices == 'CALL') / len(choices)
    put_choice_ratio = np.sum(choices == 'PUT') / len(choices)

    print(f"Underlying Option Type: {params['option_type'].upper()} Option")
    print(f"Initial Stock Price S0: {params['s0']}")
    print(f"Strike Price K: {params['K']}")
    print(f"Choice Time: {params['choice_date']} years")
    print(f"Total Time to Expiration: {params['T']} years")
    print(f"Number of Simulation Paths: {params['n_paths']:,}")
    print("-" * 70)
    print(f"Chooser Option Value: {price:.4f}")
    print(f"Standard Error: {std_error:.6f}")
    print(f"95% Confidence Interval: [{price - 1.96*std_error:.4f}, {price + 1.96*std_error:.4f}]")
    print(f"Call Choice Ratio: {call_choice_ratio:.2%}")
    print(f"Put Choice Ratio: {put_choice_ratio:.2%}")
    print(f"Average Call Option Value (at choice date): {np.mean(call_vals):.4f}")
    print(f"Average Put Option Value (at choice date): {np.mean(put_vals):.4f}")
    print("=" * 70)

    # =============================================================================
    # Visualization Analysis (English Labels)
    # =============================================================================

    plt.figure(figsize=(15, 10))

    # 1. Sample Price Paths
    plt.subplot(2, 3, 1)
    for i in range(min(50, params['n_paths'])):
        plt.plot(paths[i, :], alpha=0.1, color='blue')
    plt.axvline(x=params['choice_date']/params['T'] * params['n_steps'], 
            color='red', linestyle='--', linewidth=2, label='Choice Date')
    plt.title('Sample Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    # 2. Choice Distribution
    plt.subplot(2, 3, 2)
    choice_counts = [np.sum(choices == 'CALL'), np.sum(choices == 'PUT')]
    plt.bar(['CALL', 'PUT'], choice_counts, color=['green', 'red'], alpha=0.7)
    plt.title('Choice Distribution')
    plt.ylabel('Number of Paths')
    for i, v in enumerate(choice_counts):
        plt.text(i, v + max(choice_counts)*0.01, f'{v}\n({v/len(choices):.1%})', 
                ha='center', va='bottom')
    plt.grid(True, axis='y')

    # 3. Call vs Put Value Scatter Plot
    plt.subplot(2, 3, 3)
    plt.scatter(call_vals, put_vals, alpha=0.1, s=1)
    max_val = max(np.max(call_vals), np.max(put_vals))
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='y=x')
    plt.xlabel('Call Option Value')
    plt.ylabel('Put Option Value')
    plt.title('Value Comparison at Choice Date')
    plt.legend()
    plt.grid(True)

    # 4. Stock Price at Choice Date vs Choice
    plt.subplot(2, 3, 4)
    choice_prices = paths[:, int(params['choice_date']/params['T'] * params['n_steps'])]
    call_mask = (choices == 'CALL')
    put_mask = (choices == 'PUT')

    plt.hist(choice_prices[call_mask], bins=50, alpha=0.7, color='green', 
            label='Choose CALL', density=True)
    plt.hist(choice_prices[put_mask], bins=50, alpha=0.7, color='red', 
            label='Choose PUT', density=True)
    plt.axvline(x=params['K'], color='black', linestyle='--', label=f'Strike K={params["K"]}')
    plt.xlabel('Stock Price at Choice Date')
    plt.ylabel('Density')
    plt.title('Stock Price Distribution at Choice Date')
    plt.legend()
    plt.grid(True)

    # 5. Payoff Distribution
    plt.subplot(2, 3, 5)
    final_payoffs = np.zeros(params['n_paths'])
    choice_step = int(params['choice_date'] / params['T'] * params['n_steps'])

    for i in range(params['n_paths']):
        if choices[i] == 'CALL':
            payoff = asian_option_value(paths[i:i+1, choice_step:], params['K'], 
                                        params['r'], params['T'], 'call')
        else:
            payoff = asian_option_value(paths[i:i+1, choice_step:], params['K'], 
                                        params['r'], params['T'], 'put')
        final_payoffs[i] = payoff[0]

    plt.hist(final_payoffs, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Final Payoff Distribution')
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 6. Convergence Analysis
    plt.subplot(2, 3, 6)
    convergence_prices = []
    sample_sizes = range(1000, params['n_paths'] + 1, 5000)

    for size in sample_sizes:
        # Simplified handling, should resample in practice
        sub_payoffs = final_payoffs[:size]
        conv_price = np.exp(-params['r'] * params['choice_date']) * np.mean(sub_payoffs)
        convergence_prices.append(conv_price)

    plt.plot(sample_sizes, convergence_prices)
    plt.axhline(y=price, color='red', linestyle='--', label=f'Final Value: {price:.4f}')
    plt.title('Monte Carlo Convergence')
    plt.xlabel('Number of Simulated Paths')
    plt.ylabel('Option Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Sensitivity Analysis
    # =============================================================================

    print("\nSensitivity Analysis:")
    print("-" * 50)

    # Sensitivity to different choice dates
    choice_dates = [0.25, 0.5, 0.75]
    print("Impact of Different Choice Dates:")
    for choice_date in choice_dates:
        test_params = params.copy()
        test_params['choice_date'] = choice_date
        test_params['n_paths'] = 10000  # Reduce paths for faster computation
        
        test_price, _, test_choices, _, _, _ = path_dependent_chooser_option(**test_params)
        call_ratio = np.sum(test_choices == 'CALL') / len(test_choices)
        print(f"  Choice Date={choice_date}: Value={test_price:.4f}, Call Choice Ratio={call_ratio:.2%}")

    # Sensitivity to different strike prices
    strikes = [95, 100, 105, 110, 115]
    print("\nImpact of Different Strike Prices:")
    for strike in strikes:
        test_params = params.copy()
        test_params['K'] = strike
        test_params['n_paths'] = 10000
        
        test_price, _, test_choices, _, _, _ = path_dependent_chooser_option(**test_params)
        call_ratio = np.sum(test_choices == 'CALL') / len(test_choices)
        print(f"  K={strike}: Value={test_price:.4f}, Call Choice Ratio={call_ratio:.2%}")