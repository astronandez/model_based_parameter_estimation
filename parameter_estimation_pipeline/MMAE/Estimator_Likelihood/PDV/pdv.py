import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
import scipy.stats

from .compute_scalar_likelihood import ComputeScalarLikelihood
from .compute_model_pdv import ComputeModelPDV

class PDV:
    def __init__(self):
        # Compute scalar likelihood initialization
        self.ComputeScalarLikelihood = ComputeScalarLikelihood()

        # Compute model pdv initialization
        self.ComputeModelPDV = ComputeModelPDV()


    def update(self, r: ndarray, A: ndarray) -> float:
        q = self.ComputeScalarLikelihood.update(r, A)
        pdv = self.ComputeModelPDV.update(q, A)

        return pdv


########### Testbench ###########

def test_pdv():
    pdv_calculator = PDV()

    # Helper function to compute the PDF using scipy
    def compute_pdf_scipy(r, A):
        mean = np.zeros(A.shape[0])
        rv = scipy.stats.multivariate_normal(mean, A)
        return rv.pdf(r.flatten())

    # Simple test cases
    def simple_test_case(r, A):
        expected_pdv = compute_pdf_scipy(r, A)
        computed_pdv = pdv_calculator.update(r, A)
        print(f"Simple test case - r: {r.flatten()}, A: \n{A}")
        print(f"Expected PDV: {expected_pdv}, Computed PDV: {computed_pdv}")
        assert np.isclose(computed_pdv, expected_pdv), "Simple test case failed"
        print("Simple test case passed\n")

    r1 = np.array([[1], [0]])
    A1 = np.eye(2)
    simple_test_case(r1, A1)

    r2 = np.array([[1], [1]])
    A2 = np.array([[2, 0.5], [0.5, 1]])
    simple_test_case(r2, A2)

    # Edge case: Singular matrix A (should raise an error)
    def edge_case_test(r, A):
        try:
            pdv_calculator.update(r, A)
            print("Error: Singular matrix did not raise an error as expected")
            assert False, "Singular matrix test failed"
        except np.linalg.LinAlgError:
            print("Passed: Singular matrix raised an error as expected\n")

    r3 = np.array([[1], [1]])
    A3 = np.array([[1, 1], [1, 1]])  # Singular matrix
    edge_case_test(r3, A3)

    # Extended randomized testing with varying dimensions
    def extended_random_test_case(num_tests=100, max_dimension=10):
        np.random.seed(42)
        for i in range(num_tests):
            dim = np.random.randint(1, max_dimension + 1)
            r_random = np.random.rand(dim, 1)
            A_random = np.random.rand(dim, dim)
            A_random = A_random @ A_random.T  # Ensure A is positive definite
            computed_pdv_random = pdv_calculator.update(r_random, A_random)
            expected_pdv_random = compute_pdf_scipy(r_random, A_random)
            print(f"Random test case {i+1} - Dimension: {dim}, r: {r_random.flatten()}, A: \n{A_random}")
            print(f"Computed PDV: {computed_pdv_random}, Expected PDV: {expected_pdv_random}")
            assert np.isclose(computed_pdv_random, expected_pdv_random, rtol=1e-5), f"Randomized test failed at test case {i+1}"
            print("Random test case passed\n")

    extended_random_test_case()

# Run the test function
if __name__ == "__main__":
    test_pdv()