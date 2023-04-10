from math import sqrt

from qiskit import QuantumCircuit, transpile

from decomposer import Decomposer


class BasicDecomposer(Decomposer):
    """
    Basic decomposer.
    """

    def decompose(self, barrier=False) -> QuantumCircuit:
        """ Decomposes the matrix list into a quantum circuit. """
        n = self.get_adapter().get_matrix_count()
        circuit = QuantumCircuit(n)
        layer_list = self.get_adapter().get_layer_list()
        for layer in layer_list:
            i = 0
            for mat in layer:
                temp = self._decompose_matrix(mat, i, n, barrier)
                self._gate_list.append(temp)
                circuit = circuit.compose(temp)
                i += 1
        return circuit

    def get_circuit_list(self, barrier=False) -> list:  # not used
        """ Decomposes the matrix list into a quantum circuit. """
        i = 0
        layer_list = self.get_adapter().get_layer_list()
        circuit_list = []

        for matrix in layer_list:
            circuit_list.append(self._decompose_matrix(matrix, i, sqrt(matrix.size), barrier))
            if i == 0:
                i = 1
        return circuit_list

    def _decompose_matrix(self, matrix, index, qubits_num, barrier) -> QuantumCircuit:
        """ Decomposes a matrix into a quantum circuit. """
        circuit = QuantumCircuit(qubits_num)
        if matrix.size == 4:
            circuit.unitary(matrix, index)
        else:
            if index == 0:
                index = 1
            circuit.unitary(matrix, [index - 1, index])
        # trans_circuit = transpile(circuit, basis_gates=['cx', 'u3'])
        trans_circuit = transpile(circuit, basis_gates=['cx', 'rx', 'rz'])
        if barrier:
            trans_circuit.barrier()
        return trans_circuit

    def get_adapter(self):
        return self._mps_adapter

    def __init__(self, mps_adapter):
        super().__init__()
        self._mps_adapter = mps_adapter
        self._gate_list = []
