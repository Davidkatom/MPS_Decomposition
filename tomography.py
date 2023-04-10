"""
Create a density matrix based on the histogram, measure fidelity and purity.
"""
import itertools

import numpy as np
import random
# from qiskit.utils import tensorproduct
import copy

from qiskit.quantum_info import Pauli


def __advance_index(indexes, size, i):
    """
    Advance an N dimensional index.
    :param indexes: list of indexes
    :param size: Maximum index
    :param i: Index to advance
    :return: New list
    """
    if i == -1:
        return
    indexes[i] = indexes[i] + 1
    if indexes[i] == size:
        indexes[i] = 0
        __advance_index(indexes, size, i - 1)


def get_density_matrix(results, shots, dims):
    """
    Driver Method
    :param s: number of circuits
    :param results: Qiskit Counts obj
    :return: Density Matrix
    """
    results_copy = copy.deepcopy(results)

    # get number of qubits and shots.

    size = 4
    array_dim = tuple([size] * dims)
    array = np.zeros(array_dim)

    indexes = [0] * dims

    for cell in array.flat:
        array[(tuple(indexes))] = __calc_cell(results_copy, indexes, size, shots, [])
        __advance_index(indexes, size, dims - 1)
    return __get_matrix(array, size, dims)


def __calc_cell(results, indexes, size, shots, ignore):
    """
    Calculate the corresponding coefficient.
    :param results: Qiskit Result obj
    :param indexes: List of indexes
    :param size: Maximum index
    :param shots: Shots
    :param ignore: Indexes to ignore
    :return: Coefficient
    """

    options = [a for a in range(1, size)]
    cell = 0
    mult = 1
    if shots == 0:
        return 0

    if sum(indexes) == 0:
        return 1
    if 0 not in indexes:
        index = __get_flat_index(indexes, size)
        for x, y in results[index].items():
            j = 0
            mult = 1
            rev_x = x[::-1]
            for char in rev_x:
                j += 1
                if j - 1 in ignore:
                    continue
                mult *= pow(-1, int(char))
            cell = cell + int(mult * y)
        cell = cell / shots
        return cell
    else:
        j = 0
        for option in options:
            index_cpy = indexes.copy()
            ig_index = index_cpy.index(0)
            index_cpy[index_cpy.index(0)] = option
            ignore.append(ig_index)
            cell = cell + __calc_cell(results, index_cpy, size, shots, ignore)
            j += 1
        return cell / j


def __get_flat_index(indexes, size):
    """
    Calculates the flat index of the N-dim array.
    :param indexes: List of indexes
    :param size: Maximum Index (4)
    :return: Flat index
    """
    a = 0
    power = len(indexes) - 1
    for i in range(len(indexes)):
        a += (indexes[i] - 1) * pow(size - 1, power)
        power -= 1
    return a


def __get_matrix(array, size, dims):
    """
    Generates Matrix.
    :param array: N-dim coefficient array
    :param size: Maximum index (4)
    :param dims: array dimensions
    :return: Density matrix
    """
    identity = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigmas = [identity, sigma_x, sigma_y, sigma_z]

    indexes = [0] * dims
    matrix = np.zeros((pow(2, dims), pow(2, dims)))

    j = 0
    array_flat = []
    for cell in array.flat:
        print(cell)
        array_flat.append(cell)
        tensor = np.array([1])
        for i in range(len(indexes) - 1, -1, -1):
            tensor = np.kron(tensor, sigmas[indexes[i]])
        matrix = np.add(matrix, tensor * cell * (1 / pow(2, len(indexes))))
        __advance_index(indexes, size, dims - 1)
    # abs_array = [-1*np.absolute(a) for a in array_flat]
    # print(np.sort(abs_array))
    # print(list(-1*np.sort(abs_array)))
    return matrix


def get_expv(matrix, n_qubits):
    opt = [['I', 'X', 'Y', 'Z']] * n_qubits
    cof = {}
    for combination in itertools.product(*opt):
        pauli = Pauli(''.join(combination))
        val = np.trace(np.matmul(pauli.to_matrix(),matrix))
        cof[pauli.to_label()] = val
    return cof


def calc_purity(matrix):
    """
    Calculates purity: tr(P^2)
    :param matrix: Density matrix
    :return: purity
    """
    return np.trace(np.matmul(matrix, matrix))


def calc_fidelity(matrix, vector):
    """
    Calculate fidelity to a given vector
    :param matrix: Density matrix
    :param vector: State vector
    :return: Fidelity
    """
    return np.dot(np.dot(vector.conjugate()._data, matrix), vector._data)
