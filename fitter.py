import time
from computer_vision.tools.common import *
from scipy.optimize import curve_fit

def dampedCosine(t, m, k, b, amplitude, phi, offset):
    alpha = (b/(2 * m))
    omega = sqrt((k/m) - alpha **2)
    return abs(amplitude) * exp(-alpha * t) * cos(omega * t + phi) + offset

def createSyntheticFunction(function, params, t_max, steps):
    t = linspace(0.0, t_max, steps)
    z = function(t, *params)
    z_mu = mean(z)
    return t, z, z_mu

def fitToRealData(model_id, params, params_guess, ts, cys, store=True):
    m, k, b, amplitude, phi, offset= params
    print(f"Real Damped Cosine Values")
    print(f"m: {m} kg,  k: {k} N/m, b: {b} N/s, Amplitude: {amplitude} m, Phi: {phi}, offset: {offset}")
    
    params_fit, covariance = curve_fit(dampedCosine, ts, cys, p0=params_guess, maxfev=500000)
    m_fit, k_fit, b_fit, amplitude_fit, phi_fit, offset_fit = params_fit
    print(f"Estimated Damped Cosine Values")
    print(f"m: {m_fit} kg,  k: {k_fit} N/m, b: {b_fit} N/s, Amplitude: {amplitude_fit} m, Phi: {phi_fit}, offset: {offset_fit}")
    t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params_fit, ts[-1], len(ts) * 2)
    
    labels2 = [f"./graphs/{model_id}_fitment_{time.strftime('%Y%m%d_%H%M%S')}.fig",
                "Damped Coside Curve Fitment"]
    plotFitCurve(params, cys, ts, z_guess, t_guess, labels2, store=store)
    print("Original Ratios k/m:", k/m, "b/m:", b/m)
    print("Fitment Ratios k/m:", k_fit/m_fit, "b/m:", b_fit/m_fit)
    plt.show()

if __name__ == "__main__":
    
    def testbenchGuessParamFromSynthetic(model_id, params, params_guess, bounds, tmax, steps):
        m, k, b, amplitude, phi, offset= params

        t, z, z_mu = createSyntheticFunction(dampedCosine, [m, k, b, amplitude, phi, offset], tmax, steps)
        labels = [f"./graphs/{model_id}_SYNTH_measurements_{time.strftime('%Y%m%d_%H%M%S')}.fig", 
                f"Synthetic Function Generated",
                    "Time (s)", "Position (m)"]
        plotTimeSeries(t, z, labels)

        params_fit, covariance = curve_fit(dampedCosine, t, z, p0=params_guess, bounds=bounds)
        m_fit, k_fit, b_fit, amplitude_fit, phi_fit, offset_fit = params_fit
        print("Estimated Fitment")
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params_fit, tmax, steps * 2)
        labels2 = [f"./graphs/{model_id}_SYNTH_fitment_{time.strftime('%Y%m%d_%H%M%S')}.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, z, t, z_guess, t_guess, labels2)
        print("Original Ratios k/m:", k/m, "b/m:", b/m)
        print("Fitment Ratios k/m:", k_fit/m_fit, "b/m:", b_fit/m_fit)
        plt.show()
    
    def testbenchGuessParamFromReal(model_id, params, params_guess, bounds):
        m, k, b, amplitude, phi, offset= params
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
        print(f"Real Damped Cosine Values")
        print(f"m: {m} kg,  k: {k} N/m, b: {b} N/s, Amplitude: {amplitude} m, Phi: {phi}, offset: {offset}")
        
        # params_fit, covariance = curve_fit(dampedCosine, ts, cys, p0=params, bounds=bounds, maxfev=15000)
        params_fit, covariance = curve_fit(dampedCosine, ts, cys, p0=params_guess, maxfev=500000)
        m_fit, k_fit, b_fit, amplitude_fit, phi_fit, offset_fit = params_fit
        print(f"Estimated Damped Cosine Values")
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params_fit, ts[-1], len(ts) * 2)
        
        labels2 = [f"./graphs/{model_id}_fitment_{time.strftime('%Y%m%d_%H%M%S')}.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, cys, ts, z_guess, t_guess, labels2)
        print("Original Ratios k/m:", k/m, "b/m:", b/m)
        print("Fitment Ratios k/m:", k_fit/m_fit, "b/m:", b_fit/m_fit)
        plt.show()

    # Used during 1:1's to fit to a synthetic function
    # model_id = "m105_5_k80_80"
    # params = [0.1055, 80.80, 0.005, -0.005, -pi, 343]
    # params_guess = [0.1, 8.0, 0.01, -0.02, 0.0, 0.0]
    # bounds = ([0.01, 8.0, 0.0, -0.1, -pi, 0.0],
    #           [0.5, 100.0, 1.0, 0.1, pi, 1000.0])
    # tmax = 100
    # steps = 5000  
    # testbenchGuessParamFromSynthetic(model_id, params, params_guess, bounds, tmax, steps)
    
    model_id = "m95_0_k80_80"
    params = [0.0950, 80.80, 0.00114, 18.14944, 3.4395, 330]
    params_guess = [0.1, 10.0, 0.01, 0.02, 0.0, 0.0]
    bounds = ([0.01, 10.0, 0.0, 0, -pi, 0.0],
              [0.3, 100.0, 0.1, 50, 2 * pi, 1000.0])
    testbenchGuessParamFromReal(model_id, params, params_guess, bounds)