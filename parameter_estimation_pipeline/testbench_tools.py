import numpy as np
import json
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
    
def simulation_configuration_setup(config_path):
    config = load_config(config_path)

    m_start = config["m_variants_start"]
    m_end = config["m_variants_end"]
    m_step = config["m_variants_step"]

    k_start = config["k_variants_start"]
    k_end = config["k_variants_end"]
    k_step = config["k_variants_step"]

    b_start = config["b_variants_start"]
    b_end = config["b_variants_end"]
    b_step = config["b_variants_step"]

    # Generate model variants
    ms = np.arange(m_start, m_end + m_step, m_step).tolist()
    ks = np.arange(k_start, k_end + k_step, k_step).tolist()
    bs = np.arange(b_start, b_end + b_step, b_step).tolist()

    # Generate all possible combinations of m, k, and b
    λs = [np.array(λ) for λ in product(ms, ks, bs)]

    m = config['true_m']
    k = config['true_k']
    b = config['true_b']
    dt = config["dt"]
    H = np.array(config["H"])
    
    # Q and R for the MMAE estimator
    Q_mmae = np.eye(H.shape[1]) * config["Q_mmae"]
    R_mmae = np.eye(H.shape[0]) * config["R_mmae"]

    # Q and R for the true system
    Q_true_system = np.eye(H.shape[1]) * config["Q_true_system"]
    R_true_system = np.eye(H.shape[0]) * config["R_true_system"]

    x0 = np.array(config["initial_state"])
    max_time = config['max_time']
    max_steps = int(config['max_time'] / dt)
    amplitude = config['amplitude']
    random_seed = config['random_seed']
    true_system_noisy = config['true_system_noisy']

    return λs, m, k, b, dt, H, Q_mmae, R_mmae, Q_true_system, R_true_system, x0, max_time, max_steps, amplitude, random_seed, true_system_noisy

def return_csv(times, data, title):
    df = pd.DataFrame(data)
    df.insert(0, "Time", times)  # Insert times as the first column
    df.columns = ["Time"] + [f"Model_{i+1}" for i in range(len(df.columns) - 1)]
    df.to_csv(title, index=False)

def plot_csv_data(csv_file: str):
    """
    Reads a CSV file and plots each column as a separate line.
    Assumes the first column is 'times'.
    
    Parameters:
        csv_file (str): Path to the CSV file.
    """
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Assume the first column is 'times'
    times = data.iloc[:, 0]
    values = data.iloc[:, 1:]  # All other columns are data to plot

    # Plot each column as a separate line
    plt.figure(figsize=(10, 6))
    for col in values.columns:
        plt.plot(times, values[col], label=col)

    # Add labels, legend, and grid
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Measurements Over Time')
    plt.legend(title='Columns')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_λ_hat(times, lambda_hats, true_λ):
    # Convert the list of lambda_hats (which are vectors) to a numpy array for easy indexing
    lambda_hats = np.array(lambda_hats)

    fig, ax1 = plt.subplots()

    # Plot estimated mass (m) on the primary y-axis
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mass (m)', color='blue')
    ax1.plot(times, lambda_hats[:, 0], label='Estimated mass (m)', color='blue', linestyle='-')
    ax1.axhline(y=true_λ[0], color='blue', linestyle='--', label='True mass (m)')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for spring constant (k)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Spring constant (k)', color='green')
    ax2.plot(times, lambda_hats[:, 1], label='Estimated spring constant (k)', color='green', linestyle='-')
    ax2.axhline(y=true_λ[1], color='green', linestyle='--', label='True spring constant (k)')
    ax2.tick_params(axis='y', labelcolor='green')

    # Create another secondary y-axis for damping coefficient (b) (offset from the right)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset third axis to the right
    ax3.set_ylabel('Damping coefficient (b)', color='red')
    ax3.plot(times, lambda_hats[:, 2], label='Estimated damping coefficient (b)', color='red', linestyle='-')
    ax3.axhline(y=true_λ[2], color='red', linestyle='--', label='True damping coefficient (b)')
    ax3.tick_params(axis='y', labelcolor='red')

    # Add legends for each axis
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='lower right')

    plt.title('Parameter Estimates (m, k, b) vs Time')
    fig.tight_layout()  # To avoid overlap of labels
    plt.grid(True)
    plt.show()

def plot_heatmap(models_summary, times, λs, title):
    # Convert to string for label on plot
    λs = [f"m: {round(λ[0], 2)}, k: {round(λ[1], 2)}, b: {round(λ[2], 2)}" for λ in λs]
    plt.figure(figsize=(10, 6))
    plt.imshow(models_summary.T, aspect='auto', cmap='hot', origin='lower',
               extent=[times[0], times[-1], λs[0], λs[-1]])

    plt.colorbar(label='Model Likelihoods/Probabilities')
    plt.xlabel('Time (s)')
    plt.ylabel('Model Variants (λs)')
    plt.title(title)
    plt.show()

def mmae_simulator_plots(times, true_λ, λs, zs, lambda_hats, cumulative_posteriors_summary, pdvs_summary):
    # Convert the list of zs (measurements) to a numpy array
    zs = np.array(zs)
    
    # Convert the list of lambda_hat values (vectors) to a numpy array
    lambda_hats = np.array(lambda_hats)

    # Convert the list of model probabilities to a numpy array
    pdvs_summary = np.array(pdvs_summary)

    # Convert the list of model probabilities to a numpy array
    cumulative_posteriors_summary = np.array(cumulative_posteriors_summary)

    # Make csv ouputs
    return_csv(times, zs, title="./output/measurements.csv")
    return_csv(times, lambda_hats, title="./output/lambda_hats.csv")
    return_csv(times, cumulative_posteriors_summary, title="./output/cumulative_posteriors_summary.csv")
    return_csv(times, pdvs_summary, title="./output/pdvs_summary.csv")

    # Plot csv data
    plot_csv_data("./output/measurements.csv")

    # Plot the estimated variables over time
    plot_λ_hat(times, lambda_hats, true_λ)

    # # Plot the heatmap for likelihoods (PDVs) over time
    # plot_heatmap(pdvs_summary, times, λs, title="Heatmap of Model Likelihood Over Time")

    # # Plot the heatmap for model cumulative posterior probabilities over time
    # plot_heatmap(cumulative_posteriors_summary, times, λs, title="Heatmap of Cumulative Posterior Probabilities Over Time")