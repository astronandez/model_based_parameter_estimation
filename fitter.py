import time
from computer_vision.tools.common import *
from numpy import log, clip, exp, cos, argmax, arccos, concatenate, inf, argsort, hanning, ones_like, interp, unique, nan
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.fftpack import rfftfreq, rfft
from scipy.interpolate import interp1d, CubicSpline

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
    return A * exp(-clip(alpha * t, -700, 700))

def initialFitmentAnalysis(t, y):
    freqs, fft_data, peak_freq = frequencyAnalysis(y)
    omega_est = 2 * pi * peak_freq
    # phi_est = arccos(y[0] / max(y)) if y[0] >= 0 else -arccos(y[0] / max(y))
    # phi_est = arccos(clip(y[0] / max(y), -1.0, 1.0)) if y[0] >= 0 else -arccos(clip(y[0] / max(y), -1.0, 1.0))
    max_y = max(y)
    if max_y == 0 or abs(y[0] / max_y) > 1:
        phi_est = 0  # Default to zero if computation is invalid
    else:
        phi_est = arccos(clip(y[0] / max_y, -1.0, 1.0)) if y[0] >= 0 else -arccos(clip(y[0] / max_y, -1.0, 1.0))

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

def segmentedFitment(t, y, segment_size, overlap=0.75):
    step = int(segment_size * (1 - overlap))  # Overlapping step size
    fitted_segments = []
    t_segments = []
    weights = []
    segment_params = []  # Store parameters for each segment
    
    for start in range(0, len(t) - segment_size, step):
        end = start + segment_size
        t_seg = t[start:end]
        y_seg = y[start:end]
        
        if len(t_seg) < segment_size:
            break
        
        _, _, initial_guess = initialFitmentAnalysis(t_seg, y_seg)
        print(initial_guess)
        try:
            params_opt, _ = curve_fit(dampedCosine, t_seg, y_seg, p0=initial_guess, maxfev=10000)
            fitted_segment = dampedCosine(t_seg, *params_opt)
            weight = hanning(len(t_seg))  # Smooth weight function
            fitted_segments.append(fitted_segment * weight)
            t_segments.append(t_seg)
            weights.append(weight)
            segment_params.append(params_opt)  # Store parameters
        except RuntimeError:
            print(f"Fit did not converge for segment starting at index {start}")
            fitted_segments.append(interp(t_seg, t, y))  # Interpolate missing segment
            weights.append(ones_like(t_seg))
            segment_params.append(None)  # Store None if fit fails
    
    return t_segments, fitted_segments, weights, segment_params, initial_guess

def reconstructWave(t_segments, fitted_segments, weights):
    t_reconstructed = concatenate(t_segments)
    weighted_sum = concatenate(fitted_segments)
    weight_sum = concatenate(weights)

    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1e-6  # Small number instead of NaN
    reconstructed = weighted_sum / weight_sum  

    # Ensure sorting to avoid misalignment
    sorted_indices = argsort(t_reconstructed)
    t_reconstructed = t_reconstructed[sorted_indices]
    reconstructed = reconstructed[sorted_indices]

    # Remove duplicate time values
    unique_t, unique_indices = unique(t_reconstructed, return_index=True)
    unique_y = reconstructed[unique_indices]

    # Debugging statements
    print(f"Unique t length: {len(unique_t)}, Unique y length: {len(unique_y)}")
    print(f"Min t: {unique_t.min()}, Max t: {unique_t.max()}")

    if len(unique_t) < 2:
        print("ERROR: Not enough unique points for interpolation!")
        return unique_t, unique_y  # Return raw values to prevent failure

    # Interpolation
    interpolator = interp1d(unique_t, unique_y, kind='cubic', fill_value="extrapolate")
    num_interp_points = len(unique_t)
    t_smooth = linspace(unique_t.min(), unique_t.max(), num_interp_points)
    y_smooth = interpolator(t_smooth)


    return t_smooth, y_smooth

if __name__ == "__main__":
    from computer_vision.tools.dataloader import Dataloader
    import sys
    
    def testbenchFitment(m_act, k_act, b_act, t, y, segment_size, store=True):
        # New Batched data approach
        t_segments, fitted_segments, weights, segment_params, initial_guess = segmentedFitment(t, y, segment_size)
        t_smooth, y_smooth = reconstructWave(t_segments, fitted_segments, weights)

        valid_params = [params for params in segment_params if params is not None]

        if valid_params:
            valid_params = array(valid_params)  # Convert to NumPy array
            A_fit = mean(valid_params[:, 0])
            alpha_fit = mean(valid_params[:, 1])
            omega_fit= mean(valid_params[:, 2])
            phi_fit = mean(valid_params[:, 3])

        env_pos = exponentialDecay(t_smooth, A_fit, alpha_fit)
        env_neg = -exponentialDecay(t_smooth, A_fit, alpha_fit)
        
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

        plotFitment2(t, y, t_smooth, y_smooth, env_pos, env_neg, labels_fitted, store)
        plt.show()
    
    case_id = "m095_0_k80_80"
    m_act = 0.095
    k_act = 80.80
    b_act = 0.002887
    fps = 59.94
    segment_size = 10000  # Number of points per batched data
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

    # sys.stdout = open(f"./output/{case_id}_fitment_metrics.txt", 'w')
    testbenchFitment(m_act, k_act, b_act, t, y, segment_size)