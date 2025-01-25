#include <iostream>
#include <vector>
#include <cuda.h>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

__global__ void graphMatchingGPU(int* matrix, int* inOutMatching, int* outInMatching, int numNodes, int numMatchings) {
    int matchIdx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    if (matchIdx >= numMatchings) return;

    for (int j = 0; j < numNodes; j++) {
        bool matched = false;
        for (int l = 0; l < numNodes && !matched; l++) {
            if (matrix[j * numNodes + l] == 0) continue; // Skip if there is no edge
            // Use atomicCAS to ensure thread safety for matching
            if (atomicCAS(&inOutMatching[matchIdx * numNodes + l], -1, j) == -1) {
                outInMatching[matchIdx * numNodes + j] = l;
                matched = true;
                atomicSub(&matrix[j * numNodes + l], 1);
            }
        }
    }
}

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

// // Check how many threads we are allowed to use per block
// void checkMaxThreadsPerBlock() {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);

//     for (int device = 0; device < deviceCount; ++device) {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, device);

//         std::cout << "Device " << device << ": " << deviceProp.name << "\n";
//         std::cout << "  Maximum threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
//         std::cout << "  Maximum block dimensions: (" 
//                   << deviceProp.maxThreadsDim[0] << ", " 
//                   << deviceProp.maxThreadsDim[1] << ", " 
//                   << deviceProp.maxThreadsDim[2] << ")\n";
//         std::cout << "  Maximum grid dimensions: (" 
//                   << deviceProp.maxGridSize[0] << ", " 
//                   << deviceProp.maxGridSize[1] << ", " 
//                   << deviceProp.maxGridSize[2] << ")\n";
//         std::cout << std::endl;
//     }
// }

// Matching decomposition
void runGraphMatchingGPU(int numNodes, int k, std::string filename, bool warmup, bool printMatchings) {
    int numMatchings = k * numNodes;
    size_t matrixSize = numNodes * numNodes * sizeof(int);
    size_t mappingSize = numMatchings * numNodes * sizeof(int);

    // Allocate memory on the host
    int* h_matrix = (int*)malloc(matrixSize);
    int* h_inOutMatching = (int*)malloc(mappingSize);
    int* h_outInMatching = (int*)malloc(mappingSize);

    // if (!readMatrixFromFile(filename, h_matrix, numNodes)) {
    //     std::cerr << "Matrix loading failed. Exiting.\n";
    //     free(h_matrix);
    //     return -1;
    // }

    // Initialize a permutation matrix as an example.
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            if (j == (i+1)%numNodes){
                h_matrix[i * numNodes + j] = k*numNodes;
                // std::cout << "  Node " << i << " -> Node " << j << " traffic: " << k*numNodes << "\n";
            }
            else{
                h_matrix[i * numNodes + j] = 0;
            }
        }
    }

    std::fill_n(h_inOutMatching, numMatchings * numNodes, -1);
    std::fill_n(h_outInMatching, numMatchings * numNodes, -1);

    // Allocate memory on the device
    int* d_matrix;
    int* d_inOutMatching;
    int* d_outInMatching;
    cudaMalloc(&d_matrix, matrixSize);
    cudaMalloc(&d_inOutMatching, mappingSize);
    cudaMalloc(&d_outInMatching, mappingSize);

    // Copy data from host to device
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inOutMatching, h_inOutMatching, mappingSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outInMatching, h_outInMatching, mappingSize, cudaMemcpyHostToDevice);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    int blocks = (numMatchings + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(start);
    graphMatchingGPU<<<blocks, THREADS_PER_BLOCK>>>(d_matrix, d_inOutMatching, d_outInMatching, numNodes, numMatchings);
    cudaEventRecord(stop);

    // Synchronize and measure elapsed time
    cudaEventSynchronize(stop);
    float elapsedTimeMs;
    cudaEventElapsedTime(&elapsedTimeMs, start, stop);

    // Copy results back to host
    cudaMemcpy(h_inOutMatching, d_inOutMatching, mappingSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outInMatching, d_outInMatching, mappingSize, cudaMemcpyDeviceToHost);

    // Convert milliseconds to microseconds for better precision
    long long elapsedTimeNs = static_cast<long long>(elapsedTimeMs * 1e6);

    if (!warmup){
        std::cout << "NumToRs " << numNodes <<  " TimeNS: " << elapsedTimeNs << std::endl;
        if (printMatchings){
            std::cout << "Matchings:\n";
            for (int i = 0; i < numMatchings; i++) {
                std::cout << "Matching " << i + 1 << ":\n";
                for (int j = 0; j < numNodes; j++) {
                    int target = h_outInMatching[i * numNodes + j];
                    std::cout << "  Node " << j << " -> Node " << target << "\n";
                }
                std::cout << std::endl;
            }
        }
    }


    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_inOutMatching);
    cudaFree(d_outInMatching);

    // Free host memory
    free(h_matrix);
    free(h_inOutMatching);
    free(h_outInMatching);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    // if (argc < 3) {
    //     std::cerr << "Usage: " << argv[0] << " <numNodes> <k> [filename]\n";
    //     return -1;
    // }

    // int numNodes = std::stoi(argv[1]);
    // int k = std::stoi(argv[2]);
    // std::string filename = (argc > 3) ? argv[3] : "perm-matrix.txt";

    int k = 3;
    std::string filename = "perm-matrix.txt";

    int numExponents = 10;
    int numNodes = 2;
    
    // checkMaxThreadsPerBlock();
    
    // Warumup
    for (int i = 0; i< 100; i++)
        runGraphMatchingGPU(numNodes, k, filename, true, false);

    for (int i = 1; i <= numExponents; i++) {
        runGraphMatchingGPU(numNodes, k, filename, false, false);
        numNodes = numNodes * 2;
    }
    return 0;
}
