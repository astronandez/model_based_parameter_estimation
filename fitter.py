import time
from computer_vision.tools.common import *
from numpy import log, clip, exp, cos, argmax, arccos
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.fftpack import rfftfreq, rfft

def dampedCosine(t, amplitude, alpha, omega, phi):
    exponent = -alpha * t
    exponent = clip(exponent, -700, 700)
    return amplitude * exp(exponent) * cos(omega * t + phi)

def frequencyAnalysis(signal):
    N = len(signal)
    dt = float(1/59.94)
    fft_vals = abs(rfft(signal))  # Compute FFT magnitude
    freqs = rfftfreq(N, dt)  # Frequency bins
    dominant_freq = freqs[argmax(fft_vals[1:])]
    return freqs, fft_vals, dominant_freq

def exponentialDecay(t, A, alpha):
    return A * exp(-alpha * t)

def initialFitmentAnalysis(t, y):
    freqs, fft_data, peak_freq = frequencyAnalysis(y)
    omega_est = 2 * pi * peak_freq
    phi_est = arccos(y[0] / max(y)) if y[0] >= 0 else -arccos(y[0] / max(y))
    hilbert_sig = hilbert(y)  
    envelope = abs(hilbert_sig)
    params, _ = curve_fit(exponentialDecay, t, envelope, p0=[max(y), 0.1])
    A_est, alpha_est = params
    initial_guess = [A_est, alpha_est, omega_est, phi_est]
    return freqs, fft_data, initial_guess

def defaultFitment(case_id, m_act, k_act, b_act, t, y, store=True):
    labels_fft = [f"./graphs/{case_id}_y_fft.fig",
                "Frequency Spectrum of x(t)",
                "Frequency (Hz)",
                "Amplitude"]
    labels_fitted = [f"./graphs/{case_id}_fitted.fig",
                    'Damped Oscillations in the Y-axis',
                    'Time (s)',
                    'Postion (m)']
    
    # Create our initial guesses based on computation from dataset
    freqs, fft_data, initial_guess = initialFitmentAnalysis(t, y)
    params_opt, params_cov = curve_fit(dampedCosine, t, y, p0=initial_guess)
    A_fit, alpha_fit, omega_fit, phi_fit = params_opt
    
    # Create syntetic functions based on our fit parameters
    y_fit = dampedCosine(t, *params_opt)
    env_pos = exponentialDecay(t, A_fit, alpha_fit)
    env_neg = -exponentialDecay(t, A_fit, alpha_fit)

    # Calculate our fitment parameters of k and b based on a given m
    omega_0 = sqrt(omega_fit**2 + alpha_fit**2)  # Natural frequency
    k_fit = m_act * omega_0**2  # Spring constant
    b_fit = 2 * m_act * alpha_fit  # Damping coefficient
    
    print("\nEstimated and Fitted Parameters:")
    print(f"Estimated alpha = {initial_guess[1]:.3f}, Fitted alpha = {alpha_fit:.3f}")
    print(f"Estimated omega = {initial_guess[2]:.3f}, Fitted omega = {omega_fit:.3f}")
    print(f"Estimated phi = {initial_guess[3]:.3f}, Fitted phi = {phi_fit:.3f}")
    
    print("\nComputed System Parameters:")
    print(f"Fitted Spring Constant k = {k_fit:.6f} N/m")
    print(f"Fitted Damping Coefficient b = {b_fit:.6f} Ns/m \n")
    print(f"Actual Spring Constant k = {k_act:.6f} N/m")
    print(f"Actual Damping Coefficient b: {b_act:.6f} Ns/m (assuming m = {m_act} kg) \n")
    print("Original parameter ratios k/m:", k_act/m_act, "b/m:", b_act/m_act)
    print("Fitted parameter ratios k/m:", k_fit/m_act, "b/m:", b_fit/m_act)
    
    plotFFT(freqs, fft_data, labels_fft, store)
    plotFitment(t, y, y_fit, env_pos, env_neg, labels_fitted, store)

if __name__ == "__main__":
    from computer_vision.tools.dataloader import Dataloader
    import sys
    
    def testbenchFitment(m_act, k_act, b_act, t, y, labels_fft, labels_fitted):
        # Create our initial guesses based on computation from dataset
        freqs, fft_data, initial_guess = initialFitmentAnalysis(t, y)
        params_opt, params_cov = curve_fit(dampedCosine, t, y, p0=initial_guess)
        A_fit, alpha_fit, omega_fit, phi_fit = params_opt
        
        # Create syntetic functions based on our fit parameters
        y_fit = dampedCosine(t, *params_opt)
        env_pos = exponentialDecay(t, A_fit, alpha_fit)
        env_neg = -exponentialDecay(t, A_fit, alpha_fit)
        

        # Calculate our fitment parameters of k and b based on a given m
        omega_0 = sqrt(omega_fit**2 + alpha_fit**2)  # Natural frequency
        k_fit = m_act * omega_0**2  # Spring constant
        b_fit = 2 * m_act * alpha_fit  # Damping coefficient
        
        print("\nEstimated and Fitted Parameters:")
        print(f"Estimated alpha = {initial_guess[1]:.3f}, Fitted alpha = {alpha_fit:.3f}")
        print(f"Estimated omega = {initial_guess[2]:.3f}, Fitted omega = {omega_fit:.3f}")
        print(f"Estimated phi = {initial_guess[3]:.3f}, Fitted phi = {phi_fit:.3f}")
        
        print("\nComputed System Parameters:")
        print(f"Fitted Spring Constant k = {k_fit:.6f} N/m")
        print(f"Fitted Damping Coefficient b = {b_fit:.6f} Ns/m \n")
        print(f"Actual Spring Constant k = {k_act:.6f} N/m")
        print(f"Actual Damping Coefficient b: {b_act:.6f} Ns/m (assuming m = {m_act} kg) \n")
        print("Original parameter ratios k/m:", k_act/m_act, "b/m:", b_act/m_act)
        print("Fitted parameter ratios k/m:", k_fit/m_act, "b/m:", b_fit/m_act)

        plotFFT(freqs, fft_data, labels_fft)
        plotFitment(t, y, y_fit, env_pos, env_neg, labels_fitted)
        plt.show()
    
    case_id = "m095_0_k80_80"
    m_act = 0.095
    k_act = 80.80
    b_act = 0.00321
    fps = 59.94
    dataloader = Dataloader("./output/")
    _, _, _, cys, _, _ = dataloader.load(f"./data/{case_id}.csv")
    
    t = linspace(0, (len(cys) - 1) * (1 / fps), len(cys))
    y = (sqrt(mean(cys ** 2)) - cys) / 2000

    labels_fft = [f"./graphs/{case_id}_y_fft.fig",
                "Frequency Spectrum of x(t)",
                "Frequency (Hz)",
                "Amplitude"]
    labels_fitted = [f"./graphs/{case_id}_fitted.fig",
                    'Damped Oscillations in the Y-axis',
                    'Time (s)',
                    'Postion (m)']

    sys.stdout = open(f"./output/{case_id}_fitment_metrics.txt", 'w')
    testbenchFitment(m_act, k_act, b_act, t, y, labels_fft, labels_fitted)