import json
import csv
from numpy import loadtxt, polyfit
import matplotlib.pyplot as plt
from numpy import linspace, exp, cos, pi, histogram, diff, array
from scipy.stats import gaussian_kde
from matplotlib.font_manager import FontProperties
import pickle

# Must call plt.show() at the end of your grapher function calls
class Grapher:
    def __init__ (self):
        pass
    
    def display(self, path):
        with open(path, "rb") as file:
            fig = pickle.load(file) 
    
    def plotMeasurements(self, title, file_name, t_data, z_data):
        plt.figure(figsize=(10, 5))
        plt.plot(t_data, z_data, label="z")
        plt.title(f"{title}: Detector Measurements frame position in (y-axis)")
        plt.xlabel("Time (s)")
        plt.ylabel("position in y")
        plt.legend()
        plt.grid()
        with open(f"{file_name}_Measurements.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)
     
    def plotQTune(self, file_name, index, λ_est):
        cycle_boundaries = [i for i, val in enumerate(index) if val == 0]
        for i in range(len(cycle_boundaries)):
            # Define the start and end of each cycle
            start = cycle_boundaries[i]
            end = cycle_boundaries[i+1] if i+1 < len(cycle_boundaries) else len(index)
            
            # Extract the data for this cycle
            index_cycle = index[start:end]
            value_cycle = λ_est[start:end]
            
            # Plot the cycle
            plt.plot(index_cycle, value_cycle, label=f'Q {(i * 10000000)+100}')
            # plt.plot(index_cycle, value_cycle, label=f'Q {(i * 750000)+100}')
        plt.title("Parameter Estimation over time with different Q")
        plt.xlabel("Timestep")
        plt.ylabel("Parameter Value")
        plt.legend(loc='upper right', fontsize='x-small')
        # plt.savefig(f"{file_name}_QTune.png")
        with open(f"{file_name}_QTune.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)
        # if show:
        #     plt.show()
        # else:
        #     plt.close()

    def plotCurveFitment(self, file_name, t, real, t_fit, fit):
        plt.figure(figsize=(10, 5))
        plt.plot(t, real, '.', label="Observed Data")
        plt.plot(t_fit, fit, '-', label="Fitted Function")
        plt.xlabel("Frame Index")
        plt.ylabel("Displacement (m)")
        plt.legend()
        plt.title("Fittment of Spring Mass Damper System Oscillation Data")
        plt.grid(True)

        with open(f"{file_name}_fitment.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)

    def plotResidualPD(self, file_name, residual):
        num_bins = 100
        hist, bins = histogram(residual, bins=num_bins, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        kde = gaussian_kde(residual)
        x_vals = linspace(min(residual), max(residual), 100)
        kde_vals = kde(x_vals)

        plt.figure(figsize=(10, 5))
        plt.bar(bin_centers, hist, width=diff(bins), align='center', label='Probability Density')
        plt.plot(x_vals, kde_vals, color='orange', linewidth=2, label='KDE', zorder=3)
        plt.title('Probability Distribution of Residuals')
        # plt.xlim(-0.5, len(x_vals) - 0.5)
        plt.xlabel('Residuals')
        plt.ylabel('Occurances')
        plt.grid(True)

        with open(f"{file_name}_ResidualPD.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)

    def plot_λ_hat(self, title, file_name, times, lambda_hats, true_λ):
        # Convert the list of lambda_hats (which are vectors) to a numpy array for easy indexing
        lambda_hats = array(lambda_hats)

        fig, ax1 = plt.subplots(figsize=(10, 5))

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
        ax2.axhline(y=true_λ[1], color='green', linestyle='--', label='True spring constant (k)', linewidth=3)
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

        plt.title(f'{title}: Parameter Estimates (m, k, b) vs Time')
        fig.tight_layout()  # To avoid overlap of labels
        plt.grid(True)
        with open(f"{file_name}_Estimations.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)
        # plt.savefig(f"{file_name}_Estimations.png")
        # if show:
        #     plt.show()
        # else:
        #     plt.close()

    def plot_heatmap(self, file_name, models_summary, times, λs, title):
        λs = [f"m: {round(λ[0], 2)}, k: {round(λ[1], 2)}, b: {round(λ[2], 2)}" for λ in λs]
        plt.figure(figsize=(10, 6))
        plt.imshow(models_summary.T, aspect='auto', cmap='hot', origin='lower',
                extent=[times[0], times[-1], λs[0], λs[-1]])

        plt.colorbar(label='Model Likelihoods/Probabilities')
        plt.xlabel('Time (s)')
        plt.ylabel('Model Variants (λs)')
        plt.title(title)
        # plt.savefig(f"{file_name}_Heatmap.png")
        with open(f"{file_name}_Heatmap.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)
        # if show:
        #     plt.show()
        # else:
        #     plt.close()
        
    def plot_zs(self, file_name, times, ẑs, z):
        plt.figure(figsize=(10, 6))
        for i in range(ẑs.shape[1]):
            col = [arr[i, 0] for arr in ẑs]
            plt.plot(times, col, label=f"Model {i}")
            
        plt.plot(times, z, label='Actual', linestyle='--', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('measurement')
        plt.title('ẑs vs z')
        plt.legend(fontsize=6)
        plt.grid(True)

        with open(f"{file_name}_z_hats.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)

    def plotSpringConstant(self, file_name, x, f):
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

        with open(f"{file_name}_force.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)

    def plotGeneratedSystems(self, file_name, config_path, t, u, zs, simulators, resolution, z_actual = None):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        if z_actual is not None:
            ax1.plot(t[::resolution], z_actual, label="Actual Measurement", color='blue', linestyle='--')
            
        for i in range(len(zs)):
            arr = array(zs[i])
            ax1.plot(t[::resolution], arr[::resolution], label=f"m:{simulators[i].m:.4f}, k:{simulators[i].k:.4f}, b:{simulators[i].b:.4f}, Q:{simulators[i].Q[0,0]}, R:{simulators[i].R[0,0]}")
            # ax1.plot(t[::config["plot_every"]], arr[::config["plot_every"]])
        
        font_props = FontProperties(size=6)
        ax2 = ax1.twinx()
        ax2.plot(t[::resolution], [i[0,0] for i in u][::resolution], label="Input",  color='red', linestyle='--')
        plt.title(f"Level0 Validation Graph of {config_path}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Displacement (m)")
        ax2.set_ylabel("Force (N)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1leg = ax1.legend(loc='upper left', prop=font_props)
        plt.grid(True)
        
        with open(f"{file_name}_Generated.fig", "wb") as file:
            pickle.dump(plt.gcf(), file)
    
if __name__ == "__main__":
    grapher = Grapher()
    
    ############## Lemur Video Analysis ####################
    model_id1 = 'lemur_sticky_sport_load'
    model_id2 = 'lemur_sticky_sport2'
    # grapher.display(f"./runs/{model_id1}/{model_id1}_Measurements.fig")
    # grapher.display(f"./runs/{model_id1}/{model_id1}_upper_Measurements.fig")
    # grapher.display(f"./runs/{model_id1}/{model_id1}_lower_Measurements.fig")
    grapher.display(f"./runs/{model_id1}/{model_id1}_Estimations.fig")
    # grapher.display(f"./runs/{model_id1}/{model_id1}_Likelihood_Heatmap.fig")
    # grapher.display(f"./runs/{model_id1}/{model_id1}_Posteriors_Heatmap.fig")
    
    # grapher.display(f"./runs/{model_id2}/{model_id2}_Measurements.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_upper_Measurements.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_lower_Measurements.fig")
    grapher.display(f"./runs/{model_id2}/{model_id2}_Estimations.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_Likelihood_Heatmap.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_Posteriors_Heatmap.fig")
    
    plt.show()
    
    ############### Show Model MMAE Results ###############
    # model_id = 'M200_8g_K16_0100'
    # Q_value = 'Q_0.5542105263157895'
    # grapher.display(f"./runs/{model_id}/{model_id}_{Q_value}_Estimations.fig")
    # grapher.display(f"./runs/{model_id}/{model_id}_{Q_value}_Likelihood_Heatmap.fig")
    # grapher.display(f"./runs/{model_id}/{model_id}_{Q_value}_Posteriors_Heatmap.fig")
    
    ################ Show Validation graphs ###############
    # grapher.display("./runs/validation_tests/level0_Generated.fig")
    # grapher.display("./runs/validation_tests/level1_Generated.fig")
    # grapher.display("./runs/validation_tests/level2_Generated.fig")
    # plt.show()
    
    ################## Show Back Tire Measurement Graphs ##################
    # model_id1 = 'lemur_sport1'
    # model_id2 = 'lemur_sport2'
    # model_id3 = 'lemur_sport3'
    # model_id4 = 'lemur_suv1'
    # model_id5 = 'lemur_suv2'
    # model_id6 = 'lemur_suv3'
    # model_id7 = 'lemur_sport_full1'
    # model_id8 = 'lemur_sport_full2'
    # model_id9 = 'lemur_sport_full3'
    # model_id10 = 'lemur_suv_twopass1'
    # model_id11 = 'lemur_suv_twopass2'
    # model_id12 = 'lemur_suv_twopass3'
    
    # grapher.display(f"./runs/{model_id1}/{model_id1}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id3}/{model_id3}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id4}/{model_id4}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id5}/{model_id5}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id6}/{model_id6}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id7}/{model_id7}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id8}/{model_id8}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id9}/{model_id9}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id10}/{model_id10}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id11}/{model_id11}_front_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id12}/{model_id12}_front_tire_Measurements.fig")
    
    # grapher.display(f"./runs/{model_id1}/{model_id1}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id2}/{model_id2}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id3}/{model_id3}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id4}/{model_id4}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id5}/{model_id5}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id6}/{model_id6}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id7}/{model_id7}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id8}/{model_id8}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id9}/{model_id9}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id10}/{model_id10}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id11}/{model_id11}_back_tire_Measurements.fig")
    # grapher.display(f"./runs/{model_id12}/{model_id12}_back_tire_Measurements.fig")
    # plt.show()
    
    # ############### Show Spring Constant graphs ###############
    # grapher.display(f"./runs/spring_constant_tests/spring1_force.fig")
    # grapher.display(f"./runs/spring_constant_tests/spring2_force.fig")
    # grapher.display(f"./runs/spring_constant_tests/spring3_force.fig")
    # plt.show()
    
