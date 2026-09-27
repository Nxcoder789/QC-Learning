import numpy as np


class StatevectorSimulator:
    def __init__(self, num_qubits: int):
        """Initialize simulator with n qubits in the |0...0> state."""
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def normalize(self):
        """Ensure statevector Euclidean norm is equal to 1."""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def apply_gate(self, gate_matrix: np.ndarray, target: int):
        """Applies a 1-qubit gate to a specified target qubit."""
        tensor_state = self.state.reshape([2] * self.num_qubits)
        tensor_state = np.tensordot(gate_matrix, tensor_state, axes=([1], [target]))
        tensor_state = np.moveaxis(tensor_state, 0, target)
        self.state = tensor_state.flatten()
        self.normalize()

    def x(self, target: int):
        """Pauli-X (NOT) gate."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.apply_gate(X, target)

    def z(self, target: int):
        """Pauli-Z gate."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.apply_gate(Z, target)

    def h(self, target: int):
        """Hadamard gate."""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self.apply_gate(H, target)

    def cnot(self, control: int, target: int):
        """Controlled-NOT (CNOT) gate."""
        tensor_state = self.state.reshape([2] * self.num_qubits)
        slices = [slice(None)] * self.num_qubits
        slices[control] = 1
        tensor_state[tuple(slices)] = np.flip(
            tensor_state[tuple(slices)], axis=target
        )
        self.state = tensor_state.flatten()
        self.normalize()

    def cz(self, control: int, target: int):
        """Controlled-Z (CZ) gate."""
        tensor_state = self.state.reshape([2] * self.num_qubits)
        slices = [slice(None)] * self.num_qubits
        slices[control] = 1
        slices[target] = 1
        tensor_state[tuple(slices)] *= -1
        self.state = tensor_state.flatten()
        self.normalize()

    def entanglement_entropy(self) -> float:
        """Calculates von Neumann entanglement entropy across half-bipartition."""
        s = self.num_qubits // 2
        dim_a = 2**s
        dim_b = 2 ** (self.num_qubits - s)

        matrix = self.state.reshape((dim_a, dim_b))
        singular_values = np.linalg.svd(matrix, compute_uv=False)

        nonzero_s = singular_values[singular_values > 1e-12]
        probabilities = nonzero_s**2
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)


def grover_2qubit(marked_state: str = "11") -> np.ndarray:
    """Executes 2-qubit Grover's Search algorithm for a given target binary string."""
    sim = StatevectorSimulator(2)

    # 1. Uniform Superposition
    sim.h(0)
    sim.h(1)

    # 2. Oracle implementation for 2-qubit computational basis states
    if marked_state == "11":
        sim.cz(0, 1)
    elif marked_state == "10":
        sim.x(1)
        sim.cz(0, 1)
        sim.x(1)
    elif marked_state == "01":
        sim.x(0)
        sim.cz(0, 1)
        sim.x(0)
    elif marked_state == "00":
        sim.x(0)
        sim.x(1)
        sim.cz(0, 1)
        sim.x(0)
        sim.x(1)
    else:
        raise ValueError("marked_state must be one of '00', '01', '10', or '11'")

    # 3. Diffusion Operator
    sim.h(0)
    sim.h(1)
    sim.x(0)
    sim.x(1)
    sim.cz(0, 1)
    sim.x(0)
    sim.x(1)
    sim.h(0)
    sim.h(1)

    return sim.state


if __name__ == "__main__":
    target_state = "11"
    statevector = grover_2qubit(target_state)
    probabilities = np.abs(statevector) ** 2

    print(f"Grover's Search output statevector: {statevector}")
    print(f"Probabilities (|00>, |01>, |10>, |11>): {np.round(probabilities, 4)}")