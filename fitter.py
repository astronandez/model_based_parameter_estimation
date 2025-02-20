import time
from computer_vision.tools.common import *
from numpy import log, clip, exp, cos, argmax
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq

def dampedCosine(t, m, k, b, amplitude, phi, offset):
    alpha = (b/(2 * m))
    omega = sqrt((k/m) - alpha **2)
    exponent = -alpha * t
    exponent = clip(exponent, -700, 700)  # Clip to prevent overflow
    return abs(amplitude) * exp(exponent) * cos(omega * t + phi) + offset

def dampedCosine2(t, alpha, omega, amplitude, phi, offset):
    exponent = clip((-alpha * t), -700, 700)  # Clip to prevent overflow
    return abs(amplitude) * exp(exponent) * cos(omega * t + phi) + offset

def exponentialDecay(t, amplitude, alpha):
    return amplitude * exp(-alpha * t)

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
    from scipy.interpolate import interp1d
    from computer_vision.tools.dataloader import Dataloader
    
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
    
    def testbenchGuessParamFromReal2(model_id, m, k, b, params, params_guess, bounds):
        alpha, omega, amplitude, phi, offset= params
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
        y = sqrt(mean(cys ** 2)) - cys
        print(f"Real Damped Cosine Values")
        print(f"alpha: {alpha},  omega: {omega}, Amplitude: {amplitude}, Phi: {phi}, offset: {offset}")
        
        # params_fit, covariance = curve_fit(dampedCosine, ts, cys, p0=params, bounds=bounds, maxfev=15000)
        params_fit, covariance = curve_fit(dampedCosine, ts, y, p0=params_guess, maxfev=500000)
        alpha_fit, omega_fit, amplitude_fit, phi_fit, offset_fit = params_fit
        print(f"Estimated Damped Cosine Values")
        print(f"alpha: {alpha_fit},  omega: {omega_fit}, Amplitude: {amplitude_fit}, Phi: {phi_fit}, offset: {offset_fit}")
        
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params_fit, ts[-1], len(ts) * 2)
        labels2 = [f"./graphs/{model_id}_fitment_{time.strftime('%Y%m%d_%H%M%S')}.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, y, ts, z_guess, t_guess, labels2)
        print("Original Ratios k/m:", k/m, "b/m:", b/m)
        # print("Fitment Ratios k/m:", k_fit/m_fit, "b/m:", b_fit/m_fit)
        plt.show()

    def testbenchGuessParamFromReal(model_id, params, params_guess, bounds):
        m, k, b, amplitude, phi, offset= params
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
        y = sqrt(mean(cys ** 2)) - cys
        print(f"Real Damped Cosine Values")
        print(f"m: {m} kg,  k: {k} N/m, b: {b} N/s, Amplitude: {amplitude} m, Phi: {phi}, offset: {offset}")
        
        # params_fit, covariance = curve_fit(dampedCosine, ts, cys, p0=params, bounds=bounds, maxfev=15000)
        params_fit, covariance = curve_fit(dampedCosine, ts, y, p0=params_guess, maxfev=500000)
        m_fit, k_fit, b_fit, amplitude_fit, phi_fit, offset_fit = params_fit
        print(f"Estimated Damped Cosine Values")
        print(f"m: {m_fit} kg,  k: {k_fit} N/m, b: {b_fit} N/s, Amplitude: {amplitude_fit} m, Phi: {phi_fit}, offset: {offset_fit}")
        
        t_guess, z_guess, z_mu_guess = createSyntheticFunction(dampedCosine, params_fit, ts[-1], len(ts) * 2)
        labels2 = [f"./graphs/{model_id}_fitment_{time.strftime('%Y%m%d_%H%M%S')}.fig",
                   "Damped Coside Curve Fitment"]
        plotFitCurve(params, y, ts, z_guess, t_guess, labels2)
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
    
    def tester():
        model_id = "m95_0_k80_80"
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
        y = sqrt(mean(cys ** 2)) - cys
        positive_peaks, _ = find_peaks(y, distance=10)
        negative_peaks, _ = find_peaks(-y, distance=10)
        
        y_peaks = y[positive_peaks]
        t_peaks = ts[positive_peaks]
        T = mean(t_peaks[1:] - t_peaks[:-1])
        
        log_decrement = log(array(y_peaks[:-1]) / array(y_peaks[1:]))
        delta = mean((1/len(y_peaks)) * log_decrement)
        sig = delta / (sqrt(4 * (pi ** 2) + delta ** 2 )) 
        omega_n = (sqrt(4 * (pi ** 2) + delta ** 2 )) / T
        omega_d = omega_n * sqrt(1 - sig ** 2)
        
        # b = mean((2 * 0.095 / T) * log(array(y_peaks[:-1]) / array(y_peaks[1:])))
        
        peak_index = argmax(y_peaks)
        t_first_peak = t_peaks[peak_index]
        phase_offset = (2 * pi * t_first_peak) % (2 * pi)
        k= 80.80
        m = k / omega_d ** 2
        b = sig * (2 * sqrt(80.80 * 0.095))
        # omega = 2 * pi / T
        # alpha = (b/(2 * 0.095))
        
        # print("Calculated Period:", T)
        # print("Calculated b:", b)
        # print("Calculated omega:", omega)
        # print("Calculated alpha:", alpha)
        # print(f"Estimated phase offset: {phase_offset:.4f} rad")
        
        popt, _ = curve_fit(exponentialDecay, t_peaks, y_peaks, p0=(y_peaks[0], 0.1))
        amplitude, alpha = popt
        
        # Generate smooth times for the fitted curve
        smooth_times = linspace(min(t_peaks), max(t_peaks), 500)
        fitted_envelope = exponentialDecay(smooth_times, amplitude, alpha)
        

        # Plot the results
        # plt.plot(ts, y, label='Waveform')
        # plt.plot(ts[positive_peaks], y[positive_peaks], "bo", label='Positive Peaks')
        # # plt.plot(ts[negative_peaks], y[negative_peaks], "bo", label='Negative Peaks')
        # plt.plot(smooth_times, fitted_envelope, '-', label=f'Exponential Envelope: $A(t) = {amplitude:.2f}e^{{-{alpha:.6f}t}}$', linewidth=2)
        # plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # params = [0.0950, 80.80, b, amplitude, phase_offset, 0.0]
        # params_guess = [0.1, 10.0, 0.01, 0.02, 0.0, 0.0]
        # bounds = ([0.01, 10.0, 0.0, 0, -pi, 0.0],
        #           [0.3, 100.0, 0.1, 50, 2 * pi, 1000.0])
        # testbenchGuessParamFromReal(model_id, params, params_guess, bounds)
        
        
        print("Calculated Period:", T)
        # print("Calculated m:", m)
        print("Calculated b:", b)
        print("Calculated omega:", omega_d)
        print("Calculated alpha:", alpha)
        print(f"Estimated phase offset: {phase_offset:.4f} rad")
        
        params = [alpha, omega_d, amplitude, phase_offset, 0.0]
        params_guess = [0.004, 8.0, 16.0, 3.8, 0.0]
        bounds = ([0.01, 10.0, 0.0, 0, -pi, 0.0],
                [0.3, 100.0, 0.1, 50, 2 * pi, 1000.0])
        
        t, z, z_mu = createSyntheticFunction(dampedCosine, [alpha, 8.5, amplitude, -0.072, 0.0], ts[-1], len(ts))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(ts, y, label='Original Cirve', s=10)
        plt.plot(t, z, label='Fitted Curve', color='red')
        plt.xlabel('Time (t)')
        plt.ylabel('Position (y)')
        plt.legend()
        plt.title("test")
        plt.grid(True)
        plt.show()
        # testbenchGuessParamFromReal2(model_id, m, k, b, params, params_guess, bounds)
        
    model_id = "m095_0_k80_80"
    m = 0.095
    k = 80.80
    b = 0.0007283235233172228
    dataloader = Dataloader("./output/")
    ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
    y = sqrt(mean(cys ** 2)) - cys
    positive_peaks, _ = find_peaks(y, distance=10)
    y_peaks = y[positive_peaks]
    t_peaks = ts[positive_peaks]
    
    
    ts_uniform = linspace(ts[0], ts[-1], len(ts))
    interp_func = interp1d(ts, y, kind='cubic')
    y_uniform = interp_func(ts_uniform)
    
    Y = fft(y_uniform)
    freqs = fftfreq(len(ts_uniform), d=(ts_uniform[1] - ts_uniform[0]))

    magnitude = abs(Y)
    peak_index = argmax(magnitude[:len(magnitude)//2])
    peak_frequency = freqs[peak_index]
    omega = 2 * pi * peak_frequency
    
    log_amplitudes = log(y_peaks)
    alpha, intercept = polyfit(t_peaks, log_amplitudes, 1)
    
    print("Peak Frequency:", peak_frequency)
    print("Actual Omega:", omega)
    print("Fit Alpha:", alpha)
    
    omega_0 = sqrt(omega**2 + alpha**2)
    k_fit = m * omega**2
    b_fit = 2 * m * alpha
    
    print("Original Ratios k/m:", k/m, "b/m:", b/m)
    print("Fitment Ratios k/m:", k_fit/m, "b/m:", b_fit/m)
    
    t, z, z_mu = createSyntheticFunction(dampedCosine2, [abs(alpha), omega, 16.839088144725444, -0.072, 0.0], ts[-1], len(ts))
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:len(freqs)//2], abs(Y[:len(Y)//2]))  # Only plot the positive half
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(t_peaks, log_amplitudes, 'o', label='Log of Amplitudes')
    plt.plot(t_peaks, alpha * t_peaks + intercept, 'r-', label=f'Fit: alpha = {-alpha:.5f}')
    plt.xlabel('Time (s)')
    plt.ylabel('ln(Amplitude)')
    plt.title('Estimating Damping Constant (alpha)')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ts, y, label='Original Cirve', s=10)
    plt.plot(t, z, label='Fitted Curve', color='red')
    plt.xlabel('Time (t)')
    plt.ylabel('Position (y)')
    plt.legend()
    plt.title("test")
    plt.grid(True)
    
    plt.show()