import h5py
import numpy as np


class MPSAdapter:
    def __init__(self, layers, psi, num_of_qubits, expected_fidelity):
        self._layers = layers
        self._psi = psi
        self._layer_size = num_of_qubits
        self._expected_fidelity = expected_fidelity

    def get_layer_list(self):
        """
        Returns a list of all matrices in the file
        :return: list of matrices
        """
        return self._layers

    def get_matrix_count(self):
        """
        Returns the number of matrices in a layer
        :return: list of matrices
        """
        return self._layer_size

    def get_psi(self):
        """
        Returns the desired state vector
        :return: desired state vector
        """
        return self._psi


class Hdf5Reader:

    def __init__(self, filename):
        # self._matrix_list = []
        self._layers = []
        self._psi = []
        self._file_name = ''
        self._file = None
        self._layer_size = 0

        self._file_name = filename
        self._read_file()
        self._expected_fidelity = 0

        self._adapter = MPSAdapter(self._layers, self._psi, self._layer_size, self._expected_fidelity)

    def set_new_reader(self, matrix):
        self._layers = [[matrix]]
        self._psi = 1.0
        self._layer_size = 1

    def _read_file(self):
        self._file = h5py.File(self._file_name, 'r')
        gates = self._file['gates']
        self._psi = np.array(self._file['psi'])[0]
        for layer in gates:
            qubits = 1
            matrix_list = []
            for key in gates[layer]:
                matrix = np.matrix(gates[layer][key])
                if matrix.size == 16:
                    qubits += 1
                    #matrix = matrix[[0, 2, 1, 3]]
                matrix_list.append(matrix)
            self._layers.append(matrix_list)
            self._layer_size = qubits

    def get_adapter(self):
        """
        Returns the adapter
        :return: adapter
        """
        return self._adapter

    def get_matrix_by_name(self, index):
        """
        Returns the matrix at the given index
        :param index: index of the matrix
        :return: matrix at the given index
        """
        return self._layers[index]


if __name__ == '__main__':
    hdf5 = Hdf5Reader(
        "simplecase_twolayers2.h5")
    hdf5._read_file()
