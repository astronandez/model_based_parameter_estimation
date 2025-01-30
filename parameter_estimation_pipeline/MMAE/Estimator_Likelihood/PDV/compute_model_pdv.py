from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

class ComputeModelPDV:
    def __init__(self):
        pass

    def update(self, q: float, A: np.ndarray) -> float:
        """
        Calculate the probability density value (pdv) of the measurement z for the k-th model.
        """
        # Check if A is singular
        det_A = np.linalg.det(A)
        if det_A == 0:
            raise np.linalg.LinAlgError("Matrix A is singular and cannot be inverted.")

        # Compute the scalar Mahalanobis distance: q_k = r_k^T * A_k^-1 * r_k
        mahalanobis_distance_k = -0.5 * q
        
        k = A.shape[0]
        # Calculate the likelihood for the k-th model assuming a Gaussian distribution
        normalization_factor = np.sqrt(((2 * np.pi) ** (k)) * det_A)

        pdv = (1.0 / normalization_factor) * np.exp(mahalanobis_distance_k)

        return pdv
    

########### Testbench ###########

def manual_pdv_computation(q, A):
    det_A = np.linalg.det(A)
    mahalanobis_distance_k = -0.5 * q
    normalization_factor = np.sqrt(((2 * np.pi) ** (A.shape[0])) * det_A)
    return (1.0 / normalization_factor) * np.exp(mahalanobis_distance_k)

def test_compute_model_pdv():
    pdv_calculator = ComputeModelPDV()

    # Test case 1: Identity matrix A and q = 0
    q1 = 0.0
    A1 = np.eye(2)
    expected_pdv1 = manual_pdv_computation(q1, A1)
    computed_pdv1 = pdv_calculator.update(q1, A1)
    print(f"Test case 1 - Expected: {expected_pdv1}, Computed: {computed_pdv1}")
    assert np.isclose(computed_pdv1, expected_pdv1), "Test case 1 failed"

    # Test case 2: Non-identity matrix A and q
    q2 = 1.0
    A2 = np.array([[2, 0.5], [0.5, 1]])
    expected_pdv2 = manual_pdv_computation(q2, A2)
    computed_pdv2 = pdv_calculator.update(q2, A2)
    print(f"Test case 2 - Expected: {expected_pdv2}, Computed: {computed_pdv2}")
    assert np.isclose(computed_pdv2, expected_pdv2, rtol=1e-5), "Test case 2 failed"

    # Test case 3: Random matrix A and q
    np.random.seed(0)
    q3 = np.random.rand()
    A3 = np.random.rand(3, 3)
    A3 = A3 @ A3.T  # Making A positive definite
    computed_pdv3 = pdv_calculator.update(q3, A3)
    print(f"Test case 3 - Computed pdv: {computed_pdv3}")

    # Monte Carlo Simulation
    for i in range(100):
        q_random = np.random.rand()
        A_random = np.random.rand(3, 3)
        A_random = A_random @ A_random.T  # Ensure A is positive definite
        computed_pdv_random = pdv_calculator.update(q_random, A_random)
        assert computed_pdv_random >= 0, f"PDV should be non-negative, got {computed_pdv_random}"

    # Edge Case: Singular matrix A (should raise an error)
    q4 = 1.0
    A4 = np.array([[1, 1], [1, 1]])  # Singular matrix
    try:
        pdv_calculator.update(q4, A4)
        print("Test case 4 - Error: Singular matrix did not raise an error as expected")
    except np.linalg.LinAlgError:
        print("Test case 4 - Passed: Singular matrix raised an error as expected")

    # Plotting the PDV for various q values and a fixed A
    q_values = np.linspace(0, 10, 50)
    A_fixed = np.array([[2, 0.5], [0.5, 1]])
    pdvs = [pdv_calculator.update(q, A_fixed) for q in q_values]

    plt.figure()
    plt.plot(q_values, pdvs, 'b-', label='PDV for fixed A')
    plt.xlabel('q value')
    plt.ylabel('Probability Density Value')
    plt.title('PDV for Different q Values with Fixed A')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_compute_model_pdv()