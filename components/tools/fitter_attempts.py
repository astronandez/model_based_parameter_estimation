# # ############## Time Based Fit ##################
# def underdampedSHM(t, m, k, b, phi):
#     root1, root2 =  [(-b + cmath.sqrt(b**2 - 4*m*k)) / (2*m), (-b - cmath.sqrt(b**2 - 4*m*k)) / (2*m)]
#     fraction = Fraction(tan(phi)).limit_denominator(1000)
#     numerator = fraction.numerator
#     denominator = fraction.denominator
#     return (exp(-root1.real * t) * (numerator * cos(root1.imag * t) + sin(denominator * root1.imag * t) ))

# def genUnderdampedSHM(params, start, stop, dt):
#     m, k, b, phi = params
#     print(f"Perscribed Damped Cosine Values")
#     print(f"m: {m} kg, k: {k} N/m, b: {b} N/s, phi: {phi}")
#     t = arange(start, stop + dt, dt)
#     z = underdampedSHM(t, *params)
#     return t, z

# def springMassDamperODE(m, k, b, amplitude, t_impulse):
#     # Impulse force function
#     def force(t):
#         return amplitude if isclose(t, t_impulse, atol=1e-2) else 0.0

#     # Equation of motion
#     def motion(t, y):
#         x, v = y  # y[0] = x (displacement), y[1] = v (velocity)
#         dxdt = v
#         dvdt = (force(t) - b * v - k * x) / m
#         return [dxdt, dvdt]

#     return motion

# def spring_mass_damper(t, y, m, k, b, amplitude, t_impulse):
#     """
#     ODE for the spring-mass-damper system without gravity.
#     """
#     x, v = y
#     impulse_force = amplitude if isclose(t, t_impulse, atol=1e-6) else 0.0
#     dxdt = v
#     dvdt = (impulse_force - b * v - k * x) / m
#     return [dxdt, dvdt]

# def solve_spring_mass_damper(fit_params, fixed_params, time):
#     # Combine fit and fixed parameters
#     m, k, b = fit_params  # Parameters to be fit
#     amplitude, t_impulse = fixed_params  # Fixed parameters

#     # Initial conditions and ODE solution
#     initial_conditions = [0.0, 0.0]  # Start at rest
#     solution = solve_ivp(
#         spring_mass_damper,
#         (time[0], time[-1]),
#         initial_conditions,
#         t_eval=time,
#         args=(m, k, b, amplitude, t_impulse),
#     )
#     return solution.y[0]  # Return displacement
    
# def fit_function(t, m, k, b):
#     fixed_params = [10.0, 0.0]  # Fixed: amplitude, t_impulse
#     fit_params = [m, k, b]  # Parameters to fit
#     return solve_spring_mass_damper(fit_params, fixed_params, t)
    
# m, k, b, amplitude, t_impulse = 0.2, 16, 0.0001, -1.0, 0.0
# start, stop, dt = 0.0, 10000.0, 0.1
# t = arange(start, stop + dt, dt)
# equation = springMassDamperODE(m, k, b, amplitude, t_impulse)
# initial_conditions = [0.0, 0.0]
# solution = solve_ivp(equation, [start, stop], initial_conditions, t_eval=t)

# t = solution.t
# z = solution.y[0]
# z_v = solution.y[1]
# model_id = "M1000g_K50_0"
# grapher = Grapher()

# fig, ax1 = plt.subplots(figsize=(10, 5))
# ax2 = ax1.twinx() 
# ax1.plot(t[::10], z[::10], label='Displacement (absolute, including x_eq)', color='blue')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Displacement')
# plt.title('Hanging Spring-Mass-Damper System Under Gravity')
# plt.grid(True)
# plt.show()
    
# def testbed2():
#     initial_guess = [0.1, 10, 0.1, pi]
#     t, z = genUnderdampedSHM([2, 0.016, 1.0, pi], 0.0, 100.0, 0.01)
#     params_opt, params_cov = curve_fit(underdampedSHM, t, z, p0=initial_guess)
#     t_fit = arange(0.0, 100.0, 0.001)
#     z_fit = underdampedSHM(t_fit, *params_opt)
#     # r = z - z_fit
#     m, k, b, phi = params_opt
#     print(f"Fit Damped Cosine Values")
#     print(f"m: {m} kg, k: {k} N/m, b: {b} N/s, phi: {phi}")
#     grapher.plotMeasurements(f"./runs/synthetic_tests/{model_id}_DampedCosine", t, z)
#     grapher.plotCurveFitment(f"./runs/synthetic_tests/{model_id}_DampedCosine", t, z, t_fit, z_fit)
#     # grapher.plotResidualPD(f"./runs/synthetic_tests/{model_id}_DampedCosine", r)

#     ax2.plot(time, velocity, label='Velocity (v)', color='orange')
#     ax1.axhline(y=x_eq, color='green', linestyle='--', label='Equilibrium Position (x_eq)')
#     plt.plot(time, , color='red', linestyle='--', label='Impulse Applied')
#     ax2.set_ylabel('Velocity')

#     ax2.legend()
    
# def testbed3():
#     start, stop, dt = 0.0, 100.0, 0.1
#     true_params = [2.0, 16.0, 5.0]  # True values for m, k, b
#     t = arange(start, stop + dt, dt)
#     z = fit_function(t, *true_params)

#     # Fit the model to the data
#     initial_guess = [1.0, 5.0, 1.0]  # Initial guesses for m, k, b
#     bounds = ([1.0, 1.0, 0.0], [3.0, 50.0, 10.0])  # Bounds for m, k, b
#     optimal_params, covariance = curve_fit(fit_function, t, z, p0=initial_guess, bounds=bounds)

#     z_fit = fit_function(t, *optimal_params)
#     param_names = ['m', 'k', 'b']
#     print("Fitted Parameters:")
#     for name, value in zip(param_names, optimal_params):
#         print(f"{name}: {value:.4f}")

#     plt.figure(figsize=(12, 6))
#     plt.plot(t, z, label='True Curve', linestyle='-', linewidth=2)
#     plt.plot(t, z_fit, label='Fitted Curve', linestyle='--', linewidth=2)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Displacement (m)')
#     plt.title('Fit of Spring-Mass-Damper System (Fitting m, k, b)')
#     plt.legend()
#     plt.grid()
#     plt.show()
    
    