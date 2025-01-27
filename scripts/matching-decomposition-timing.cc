#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <stdint.h>
#include <immintrin.h>
#include <xmmintrin.h>

// Single-threaded graph matching
// void graphMatchingCPU(std::vector<int>& matrix, std::vector<int>& inOutMatching, std::vector<int>& outInMatching, int numNodes, int numMatchings) {
//     for (int matchIdx = 0; matchIdx < numMatchings; ++matchIdx) {
//         for (int j = 0; j < numNodes; ++j) {
//             bool matched = false;
//             // __builtin_prefetch(&outInMatching[matchIdx * numNodes + j], 1, 1); // Write intent
//             for (int l = 0; l < numNodes && !matched; ++l) {
//                 int& cell = matrix[j * numNodes + l];  // Cache reference to matrix cell
//                 // __builtin_prefetch(&inOutMatching[matchIdx * numNodes + l], 1, 1); // Write intent
//                 if (cell == 0) continue;
//                 if (inOutMatching[matchIdx * numNodes + l] == -1) {
//                     inOutMatching[matchIdx * numNodes + l] = j;
//                     outInMatching[matchIdx * numNodes + j] = l;
//                     --cell;
//                     matched = true;
//                 }
//             }
//         }
//     }
// }


void runGraphMatchingCPU(int numNodes, int k, bool warmup, bool printMatchings) {
    int numMatchings = k * numNodes;
    size_t matrixSize = numNodes * numNodes;
    size_t mappingSize = numMatchings * numNodes;

    // Allocate memory
    std::vector<int16_t> matrix(matrixSize, 0);
    std::vector<bool> inOutMatching(mappingSize, false);
    std::vector<int> outInMatching(mappingSize, -1);

    // Initialize the matrix
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            if (j == (i + 1) % numNodes) {
                matrix[i * numNodes + j] = k * numNodes;
            }
        }
    }

    // Timing
    auto start = std::chrono::high_resolution_clock::now();

    // graphMatchingCPU(matrix, inOutMatching, outInMatching, numNodes, numMatchings);
    for (int matchIdx = 0; matchIdx < numMatchings; ++matchIdx) {
        for (int j = 0; j < numNodes; ++j) {
            bool matched = false;
            // __builtin_prefetch(&outInMatching[matchIdx * numNodes + j], 1, 1); // Write intent
            for (int l = 0; l < numNodes && !matched; ++l) {
                // int& cell = matrix[j * numNodes + l];  // Cache reference to matrix cell
                // __builtin_prefetch(&inOutMatching[matchIdx * numNodes + l], 1, 1); // Write intent
                if (matrix[j * numNodes + l] == 0) continue;
                if (inOutMatching[matchIdx * numNodes + l] == false) {
                    inOutMatching[matchIdx * numNodes + l] = true;
                    outInMatching[matchIdx * numNodes + j] = l;
                    --matrix[j * numNodes + l];
                    matched = true;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (!warmup) {
        long long elapsedTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << numNodes << " " << elapsedTimeNs << " ns" << std::endl;

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
}

int main(int argc, char* argv[]) {
    int k = 3;
    int numExponents = 9;
    int numNodes = 2;

    // system("taskset -cp 2 $PPID > /dev/null 2>&1");

    for (int i = 1; i <= numExponents; ++i) {
        // warmup
        for (int j = 0; j < 8; ++j) {
            runGraphMatchingCPU(numNodes, k, true, false);
        }
        runGraphMatchingCPU(numNodes, k, false, false);
        numNodes *= 2;
    }

    return 0;
}
