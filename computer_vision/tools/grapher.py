from numpy import loadtxt, polyfit
import matplotlib.pyplot as plt
from numpy import linspace, exp, cos, pi, histogram, diff, array, mean, sqrt
from pandas import read_csv
from scipy.stats import gaussian_kde
from seaborn import histplot, kdeplot
from scipy.stats import norm
from matplotlib.font_manager import FontProperties
import pickle

def display(path):
    with open(path, "rb") as file:
        fig = pickle.load(file)

def plotTimeSeries(t_data, z_data, labels, store=True):
    file_name, title, xlabel, ylabel = labels
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_data, z_data, label="z")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if store:
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)
        
def plotValidateSyntheticSystems(model:list, sim_zs, t, u, resolution, labels, store=True):
    file_name, title, xlabel, ylabel1, ylabel2 = labels
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.title(title)
    font_props = FontProperties(size=6)
    ax2 = ax1.twinx()
    
    for i in range(len(sim_zs)):
        z_k = array(sim_zs[i])
        ax1.plot(t[::resolution], z_k[::resolution], label=f"m:{model[i].m:.4f}, k:{model[i].k:.4f}, b:{model[i].b:.4f}, Q:{model[i].Q[0,0]}, R:{model[i].R[0,0]}")
        
    ax2.plot(t[::resolution], [i[0] for i in u][::resolution], label="Input",  color='red', linestyle='--')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax2.set_ylabel(ylabel2, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1),prop=font_props)

    # Adjust plot size to make space for the legend
    plt.subplots_adjust(right=0.75)# Make space for the legen
    plt.grid(True)
    if store:
        with open(f"{file_name}", "wb") as file:
            pickle.dump(plt.gcf(), file)
        
def plotLambdaHat(t, lambda_hats, true_lambda, labels, store=True):
    # Convert the list of lambda_hats (which are vectors) to a numpy array for easy indexing
    file_name, title = labels
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot estimated mass (m) on the primary y-axis
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mass (m)', color='blue')
    ax1.plot(t, lambda_hats[:, 0], label='Estimated mass (m)', color='blue', linestyle='-')
    ax1.axhline(y=true_lambda[0], color='blue', linestyle='--', label='True mass (m)')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for spring constant (k)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Spring constant (k)', color='green')
    ax2.plot(t, lambda_hats[:, 1], label='Estimated spring constant (k)', color='green', linestyle='-')
    ax2.axhline(y=true_lambda[1], color='green', linestyle='--', label='True spring constant (k)', linewidth=3)
    ax2.tick_params(axis='y', labelcolor='green')

    # Create another secondary y-axis for damping coefficient (b) (offset from the right)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset third axis to the right
    ax3.set_ylabel('Damping coefficient (b)', color='red')
    ax3.plot(t, lambda_hats[:, 2], label='Estimated damping coefficient (b)', color='red', linestyle='-')
    ax3.axhline(y=true_lambda[2], color='red', linestyle='--', label='True damping coefficient (b)')
    ax3.tick_params(axis='y', labelcolor='red')

    # Add legends for each axis
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.legend(loc='lower right')

    plt.title(title)
    fig.tight_layout()  # To avoid overlap of labels
    plt.grid(True)
    
    if store:
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)

def plotSpringConstant(file_name, x, f, store=True):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, f, label="force")
    slope, intercept = polyfit(x, f, 1)
    trendline = slope * x + intercept
    plt.plot(x, trendline, color='red', label=f'Trendline: y = {slope:.6f}x + {intercept:.6f}')
    plt.xlabel("Displacement (m)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.title("Force vs Displacement")
    plt.grid(True)
    if store:
        with open(f"{file_name}", "wb") as file:
            pickle.dump(plt.gcf(), file)

def plotHeatmap(models_summary, ts, λs, labels, store=True):
    file_name, title = labels
    λs = [f"m: {round(λ[0], 2)}, k: {round(λ[1], 2)}, b: {round(λ[2], 2)}" for λ in λs]
    plt.figure(figsize=(10, 6))
    plt.imshow(models_summary.T, aspect='auto', cmap='hot', origin='lower',
            extent=[ts[0], ts[-1], λs[0], λs[-1]])

    plt.colorbar(label='Model Likelihoods/Probabilities')
    plt.xlabel('Time (s)')
    plt.ylabel('Model Variants (λs)')
    plt.title(title)
    if store:
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)
        
def plotFitCurve(z_orig, t_orig, z_fit, t_fit, labels, store=True):
    file_name, title = labels
    plt.figure(figsize=(10, 6))
    plt.scatter(t_orig, z_orig, label='Original Cirve', s=10)
    plt.plot(t_fit, z_fit, label='Fitted Curve', color='red')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (y)')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    if store:
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)

def plotDistribution(data, labels, store=True):
    file_name, title, xlabel, ylabel = labels
    data_mean_zero = data - mean(data)
    mu, sigma = norm.fit(data_mean_zero)
    plt.figure(figsize=(10, 6))
    histplot(data_mean_zero, bins=50, color="dodgerblue", stat="density", label="Data")
    kde = kdeplot(data_mean_zero, color='red', linestyle='-', label='KDE', linewidth=2)
    x_kde = kde.get_lines()[0].get_xdata()
    y_kde = kde.get_lines()[0].get_ydata()

    # Calculate the mean and std of the KDE
    mean_kde = sum(x_kde * y_kde) / sum(y_kde)
    std_kde = sqrt(sum((x_kde - mean_kde)**2 * y_kde) / sum(y_kde))
    plt.axvline(mean_kde, color='black', linestyle='--', label=f'Mean (KDE) = {mean_kde:.4f}')
    plt.axvline(mean_kde + std_kde, color='green', linestyle='--', label=f'Std (KDE) = {std_kde:.4f}')
    plt.axvline(mean_kde - std_kde, color='green', linestyle='--')
    x = linspace(min(data_mean_zero), max(data_mean_zero), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), color='blue', label=f'Fit: $\\mu$={mu:.4f}, $\\sigma$={sigma:.4f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if store:    
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)

def plotFFT(freqs, fft_data, labels, store=True):
    file_name, title, xlabel, ylabel = labels
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_data, label="FFT of x(t)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if store:    
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)

def plotFitment(t, y, y_fit, env_pos, env_neg, labels, store=True):
    file_name, title, xlabel, ylabel = labels
    plt.figure(figsize=(10, 6))
    plt.scatter(t, y, label='Noisy Data (y)', color='blue', s=10)
    plt.plot(t, y_fit, label='Fitted Curve (y)', color='red', linewidth=2)
    plt.plot(t, env_pos, label=f"Envelope (+)", color="green", linewidth=2)
    plt.plot(t, env_neg, label=f"Envelope (-)", color="green", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if store:    
        with open(file_name, "wb") as file:
            pickle.dump(plt.gcf(), file)   

if __name__ == "__main__":
    def loadCompleteGraphs(model_id):
        display(f"./graphs/{model_id}_x_measurements.fig")
        display(f"./graphs/{model_id}_y_measurements.fig")
        display(f"./graphs/{model_id}_width_measurements.fig")
        display(f"./graphs/{model_id}_height_measurements.fig")
        display(f"./graphs/{model_id}_x_distribution.fig")
        display(f"./graphs/{model_id}_y_distribution.fig")
        display(f"./graphs/{model_id}_width_distribution.fig")
        display(f"./graphs/{model_id}_height_distribution.fig")
        # display(f"./graphs/{model_id}_likelyhoods.fig")
        # display(f"./graphs/{model_id}_posteriors.fig")
        # display(f"./graphs/{model_id}_estimations.fig")
    
    ############### Show Validation Graph ###############
    # display(f"./data/level0.fig")
    # display(f"./data/level1.fig")
    # display(f"./data/level2.fig")
    
    ############### Show Spring Detection Graph ###############
    # loadModelGraphs("m095_0_k80_80_point_0")
    # loadModelGraphs("noise_test_point_0")
    loadCompleteGraphs("sport_point_0")
    loadCompleteGraphs("sport_point_1")
    loadCompleteGraphs("sport_point_2")
    loadCompleteGraphs("sport_point_3")
    # loadModelGraphs("m095_0_k80_80_2D")
    plt.show()
    
    ############### Show Spring Constant Graphs ###############
    # df1 = read_csv("./data/spring_constant_tests/spring1.csv", skiprows=3)
    # force1 = df1.iloc[:, 0].dropna().values
    # distance1 = df1.iloc[:, 1].dropna().values / 1000
    # df2 = read_csv("./data/spring_constant_tests/spring2.csv", skiprows=3)
    # force2 = df2.iloc[:, 0].dropna().values
    # distance2 = df2.iloc[:, 1].dropna().values / 1000
    # df3 = read_csv("./data/spring_constant_tests/spring3.csv", skiprows=3)
    # force3 = df3.iloc[:, 0].dropna().values
    # distance3 = df3.iloc[:, 1].dropna().values / 1000
    # plotSpringConstant("./data/spring_constant_tests/spring1_force.fig", distance1, force1)
    # plotSpringConstant("./data/spring_constant_tests/spring2_force.fig", distance2, force2) 
    # plotSpringConstant("./data/spring_constant_tests/spring3_force.fig", distance3, force3)   
    # display("./data/spring_constant_tests/spring1_force.fig")
    # display("./data/spring_constant_tests/spring2_force.fig")
    # display("./data/spring_constant_tests/spring3_force.fig")
    # plt.show()