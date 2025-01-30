from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

class ComputeScalarLikelihood:
    def __init__(self):
        pass


    def update(self, r: ndarray, A: ndarray) -> float:
        """
        Calculate the scalar likelihood of the observed measurement z for the k-th model.
        This is not strictly a probability but rather returns a scalar value proportional to 
        probability desnity value (pdv) of the measurement given the model.
        """
        # Calculate the inverse of A_k
        inv_A = np.linalg.inv(A)

        #Likelihood calculation
        q = r.T @ inv_A @ r
        
        return q.item()
    

########### Testbench ###########

def test_compute_scalar_likelihood():
    likelihood_calculator = ComputeScalarLikelihood()

    # Test case 1: Identity matrix A and unit vector r
    r1 = np.array([[1], [0]])
    A1 = np.eye(2)
    expected_likelihood1 = 1.0
    computed_likelihood1 = likelihood_calculator.update(r1, A1)
    print(f"Test case 1 - Expected: {expected_likelihood1}, Computed: {computed_likelihood1}")
    assert np.isclose(computed_likelihood1, expected_likelihood1), "Test case 1 failed"

    # Test case 2: Non-identity matrix A and unit vector r
    r2 = np.array([[1], [1]])
    A2 = np.array([[2, 1], [1, 2]])
    inv_A2 = np.linalg.inv(A2)
    expected_likelihood2 = (r2.T @ inv_A2 @ r2).item()
    computed_likelihood2 = likelihood_calculator.update(r2, A2)
    print(f"Test case 2 - Expected: {expected_likelihood2}, Computed: {computed_likelihood2}")
    assert np.isclose(computed_likelihood2, expected_likelihood2, rtol=1e-5), "Test case 2 failed"

    # Test case 3: Random matrix A and vector r
    np.random.seed(0)
    r3 = np.random.rand(3, 1)
    A3 = np.random.rand(3, 3)
    A3 = A3 @ A3.T  # Making A positive definite
    computed_likelihood3 = likelihood_calculator.update(r3, A3)
    print(f"Test case 3 - Computed likelihood: {computed_likelihood3}")

    # Monte Carlo Simulation
    for i in range(100):
        r_random = np.random.rand(3, 1)
        A_random = np.random.rand(3, 3)
        A_random = A_random @ A_random.T  # Ensure A is positive definite
        computed_likelihood_random = likelihood_calculator.update(r_random, A_random)
        assert computed_likelihood_random >= 0, f"Likelihood should be non-negative, got {computed_likelihood_random}"

    # Edge Case: Singular matrix A (should raise an error)
    r4 = np.array([[1], [1]])
    A4 = np.array([[1, 1], [1, 1]])  # Singular matrix
    try:
        likelihood_calculator.update(r4, A4)
        print("Test case 4 - Error: Singular matrix did not raise an error as expected")
    except np.linalg.LinAlgError:
        print("Test case 4 - Passed: Singular matrix raised an error as expected")

    # Plotting the likelihood for various r and a fixed A
    r_values = [np.array([[1], [0]]), np.array([[0], [1]]), np.array([[1], [1]]), np.array([[2], [1]])]
    A_fixed = np.array([[2, 0.5], [0.5, 1]])
    likelihoods = [likelihood_calculator.update(r, A_fixed) for r in r_values]

    plt.figure()
    for i, r in enumerate(r_values):
        plt.plot(i, likelihoods[i], 'bo', label=f'r={r.flatten()}' if i == 0 else "")

    plt.xlabel('Test Case Index')
    plt.ylabel('Likelihood')
    plt.title('Likelihood for Different r Vectors with Fixed A')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_compute_scalar_likelihood()