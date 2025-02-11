from components.tools.common import *
from scipy.optimize import curve_fit

def dampedCosine(t, m, k, b, amplitude, phi, offset):
    alpha = (b/(2 * m))
    omega = sqrt((k/m) - alpha **2)
    return abs(amplitude) * exp(-alpha * t) * cos(omega * t + phi) + offset

def createSyntheticFunction(function, params, t_max, steps):
    m, k, b, amplitude, phi, offset = params
    print(f"Perscribed Damped Cosine Values")
    print(f"m: {m} kg,  k: {k} N/m, b: {b} N/s, Amplitude: {amplitude} m, Phi: {phi}, offset: {offset}")
    t = linspace(0.0, t_max, steps)
    z = function(t, *params)
    z_mu = mean(z)
    return t, z, z_mu

if __name__ == "__main__":
    
    def testbenchGuessParamFromSynthetic(params, params_guess, tmax, steps):
        m, k, b, amplitude, phi, offset= params
        # m_g, k_g, b_g, amplitude_g, phi_g, offset_g = params_guess
        model_id = "m105_5_k80_80_SYNTH"
        t, z, z_mu = createSyntheticFunction(dampedCosine, [m, k, b, amplitude, phi, offset], tmax, steps)
        labels = [f"./graphs/{model_id}_measurements.fig", 
                f"Synthetic Function Generated",
                    "Time (s)", "Position (m)"]
        plotTimeSeries(t, z, labels)

        bounds = (
            [0.01, 10.0, 0.0, -0.1, -pi, 0.0],
            [0.5, 100.0, 0.1, 0.1, pi, 640.0]  
        )
        
        # initial_guess = [m_g, k_g, b_g, amplitude_g, phi_g, offset_g]  # Initial guess for (m, k, b, amplitude, phi, offset)

        # Perform the curve fitting
        params, covariance = curve_fit(dampedCosine, t, z, p0=params_guess, bounds=bounds)
        m_fit, k_fit, b_fit, amplitude_fit, phi_fit, offset_fit = params
        print("Estimated Fitment")
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params, tmax, steps * 2)
        labels2 = [f"{model_id}_fitment.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, z, t, z_guess, t_guess, labels2)

        print("Original Ratios k/m:", k/m, "b/m:", b/m)
        print("Guessed Ratios k/m:", k_fit/m_fit, "b/m:", b_fit/m_fit)
        plt.show()
    
    def testbenchGuessParamFromReal(model_id):
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
        
        bounds = (
            [0.01, 5.0, 0.000001, -1, -pi, 0.0],
            [0.5, 100.0, 0.1, 1, pi, 640.0]  
        )
        
        initial_guess = [0.1, 50.0, 0.001, 0.5, 0.0, 0.0]  # Initial guess for (m, k, b, amplitude, phi, offset)
        
        params, covariance = curve_fit(dampedCosine, ts, cys, p0=initial_guess, bounds=bounds)
        print("Estimated Fitment")
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params, ts[-1], len(ts))
        labels2 = [f"{model_id}_fitment.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, cys, ts, z_guess, t_guess, labels2)
        plt.show()

    params = [0.1055, 80.80, 0.006, -0.005, -pi, 343]
    params_guess = [0.1, 10.0, 0.01, -0.02, 0.0, 0.0]
    tmax = 100
    steps = 5000  
    testbenchGuessParamFromSynthetic(params, params_guess, tmax, steps)
    # testbenchGuessParamFromReal("m105_7_k80_80")