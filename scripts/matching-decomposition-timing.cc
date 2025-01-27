#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <omp.h>
#include <chrono>

// Function to read the matrix from a file
bool readMatrixFromFile(const std::string& filename, int* matrix, int numNodes) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int row = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int col = 0;
        int value;

        while (iss >> value) {
            if (col >= numNodes || row >= numNodes) {
                std::cerr << "Error: Matrix in file exceeds expected dimensions" << std::endl;
                return false;
            }
            matrix[row * numNodes + col] = value;
            col++;
        }

        if (col != numNodes) {
            std::cerr << "Error: Matrix row " << row << " has incorrect number of columns" << std::endl;
            return false;
        }

        row++;
    }

    if (row != numNodes) {
        std::cerr << "Error: Matrix file has incorrect number of rows" << std::endl;
        return false;
    }

    return true;
}

// MPI-based graph matching
void graphMatchingCPU(int* matrix, int* inOutMatching, int* outInMatching, int numNodes, int numMatchings, int rank, int size) {
    int rowsPerProcess = numNodes / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? numNodes : startRow + rowsPerProcess;

    #pragma omp parallel for
    for (int matchIdx = 0; matchIdx < numMatchings; ++matchIdx) {
        for (int j = startRow; j < endRow; ++j) {
            bool matched = false;
            for (int l = 0; l < numNodes && !matched; ++l) {
                if (matrix[j * numNodes + l] == 0) continue;
                #pragma omp critical
                {
                    if (inOutMatching[matchIdx * numNodes + l] == -1) {
                        inOutMatching[matchIdx * numNodes + l] = j;
                        outInMatching[matchIdx * numNodes + j] = l;
                        matched = true;
                        matrix[j * numNodes + l]--;
                    }
                }
            }
        }
    }

    // Synchronize the results across processes
    MPI_Allreduce(MPI_IN_PLACE, inOutMatching, numMatchings * numNodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, outInMatching, numMatchings * numNodes, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

void runGraphMatchingCPU(int numNodes, int k, const std::string& filename, bool warmup, bool printMatchings) {
    int numMatchings = k * numNodes;
    size_t matrixSize = numNodes * numNodes * sizeof(int);
    size_t mappingSize = numMatchings * numNodes * sizeof(int);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory on the host
    int* matrix = new int[numNodes * numNodes];
    int* inOutMatching = new int[numMatchings * numNodes];
    int* outInMatching = new int[numMatchings * numNodes];

    if (matrix == nullptr || inOutMatching == nullptr || outInMatching == nullptr) {
        std::cerr << "Error: Memory allocation failed on rank " << rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize the matrix
    if (rank == 0) {
        for (int i = 0; i < numNodes; ++i) {
            for (int j = 0; j < numNodes; ++j) {
                if (j == (i + 1) % numNodes) {
                    matrix[i * numNodes + j] = k * numNodes;
                } else {
                    matrix[i * numNodes + j] = 0;
                }
            }
        }
        std::fill_n(inOutMatching, numMatchings * numNodes, -1);
        std::fill_n(outInMatching, numMatchings * numNodes, -1);
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(matrix, numNodes * numNodes, MPI_INT, 0, MPI_COMM_WORLD);

    // Timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    graphMatchingCPU(matrix, inOutMatching, outInMatching, numNodes, numMatchings, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0 && !warmup) {
        long long elapsedTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << numNodes << " " << elapsedTimeNs << std::endl;

        if (printMatchings) {
            std::cout << "Matchings:\n";
            for (int i = 0; i < numMatchings; ++i) {
                std::cout << "Matching " << i + 1 << ":\n";
                for (int j = 0; j < numNodes; ++j) {
                    int target = outInMatching[i * numNodes + j];
                    std::cout << "  Node " << j << " -> Node " << target << "\n";
                }
                std::cout << std::endl;
            }
        }
    }

    delete[] matrix;
    delete[] inOutMatching;
    delete[] outInMatching;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int k = 3;
    std::string filename = "perm-matrix.txt";

    int numExponents = 10;
    int numNodes = 2;

    for (int i = 0; i < 100; ++i) {
        runGraphMatchingCPU(numNodes, k, filename, true, false);
    }

    for (int i = 1; i <= numExponents; ++i) {
        runGraphMatchingCPU(numNodes, k, filename, false, false);
        numNodes *= 2;
    }

    MPI_Finalize();
    return 0;
}
