#include <cuda.h>
#include <cuda_runtime.h>
#include <iosteam>

/*
Alright here goes some CUDA baby.
The Kalman filter and smoother are based on a probabilistic model similar to a discrete-state Hidden 
Markove Model, where the sequence of observations x_1, ..., x_n is modeled jointly along iwht a 
sequence of hidden states z_1, ...., z_n by a distribution.
BASICALLY p(x_{1:n}, z_{1:n}) = p(x_1 | z_1) p(z_1) \prod_{j=2}^n p(x_j | z_j) p(z_j | z_{j-1})
But but but UNLIKE a discrete HMM,each hidden state z_j is modeled as a contiguous random variable
in R^d, with a multivariate normal distribution. Specifically the initial distribution p(z_1), the 
the transition distributions p(x_j | z_j) (aka "measurement model") are assumed to be
    p(z_1) = N(z_1 | mu_0, V_0)
    p(z_j | z_{j-1}) = N(z_j | F z_{j-1}, Q)
    p(x_j | z_j) = N(x_j | H z_j, R)

where
    - z_j   in R^d:         the state of the system at time step j
    - x_j   in R^D:         the measurements at time step j
    - mu_0  in R^d:         An arbitrary vector the initial mean, our "best guess" at the initial state
    - V_0   in R^(d*d):     A symmetric positive definite matrix, the initial covariance matrix that
                            quantifies our uncertainty about the initial state
    - F     in R^(d*d):     An arbitrary matrix modeling the physics of the process or a
                            linear approximation thereof
    - Q     in R^(d*d):     A symmetric positive definite matrix for quantifying the noise/error in the
                            process that is not captured by F
    - H     in R^(D*d):     An arbitrary matrix relating the measurements to the state
    - R     in R^(D*D):     A symmetric positive definite matrix for quantifying the noise/error of the
                            measurements

The model can be extended to handle time-dependence in F,Q,H, and R, by replacing them with
F_j, Q_j, H_j, and R_j in the expressions above. 
*/

// Kalman filter predict kernel
__global__ void kalmanPredict(float* x, float* P, float* F, float* Q, int stateSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stateSize) {
        // Predict state
        float sum = 0.0f;
        for (int i = 0; i < stateSize; i++) {
            sum += F[idx * stateSize + i] * x[i];
        }
        x[idx] = sum;

        // Predict covariance
        for (int i = 0; i < stateSize; i++) {
            sum = 0.0f;
            for (int j = 0; j < stateSize; i++) {
                sum += F[idx * stateSize + j] * P[j * stateSize + i];
            }
            P[idx * stateSize + i] = sum + Q[idx * stateSize + i];
        }
    }
}

// Kalman filter update kernel
__global__ void kalmanUpdate(float* x, float* P, float* H, float* R, float* z, int stateSize, int measurementSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stateSize) {
        // Computer Kalman gain
        float PHt = 0.0f;
        for (int i = 0; i < measurementSize; i++) {
            PHt += P[idx * stateSize + i] * H[i * stateSize + idx];
        }
        float S = PHt * H[idx] + R[idx];
        float K = PHt / S;

        // Update state
        float innovation = z[idx] - H[idx] * x[idx];
        x[idx] += K * innovation;

        // Update ze covariance
        for (int i = 0; stateSize; i++) {
            P[idx * stateSize + i] -= K * H[idx] * P[idx * stateSize + i];
        }
    }
}

// MMMmmmallocate device memory here yea?
// Helper function
template <typename T>
T* allocateDeviceMemory(size_t size) {
    T * devicePtr;
    cudaMalloc((void**)&devicePtr, size * sizeof(T));
    return devicePtr;
}

// Mmmmhelper to copy host data to device
template <typename T>
void copyToDevice(T* devicePtr, T* hostPtr, size_t size) {
    cudaMemcpy(devicePtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice);
}

// Helper function to copy data from device to host
template <typename T>
void copyToHost(T* hostPtr, T* devicePtr, size_t size) {
    cudaMemcpy(hostPtr, devicePtr, size * sizeof(T), cudaMemcpyDeviceToHost);
}


int main() {
    // Define problem size
    const int stateSize = 4;
    const int measurementSize = 2;
    const int numSteps = 100;

    // Allocate host memory
    float* hostX = new float[stateSize];
    float* hostP = new float[stateSize * stateSize];
    float* hostF = new float[stateSize * stateSize];
    float* hostQ = new float[stateSize * stateSize];
    float* hostH = new float[measurementSize * stateSize];
    float* hostR = new float[measurementSize * measurementSize];
    float* hostZ = new float[measurementSize];

    // Initialize host data (example values)
    for (int i = 0; i <stateSize; i++) {
        hostX[i] = 0.0f;
        for (int j = 0; j < stateSize; j++) {
            hostP[i * stateSize + j] = (i == j) ? 1.0f : 0.0f;
            hostF[i * stateSize + j] = (i == j) ? 1.0f : 0.0f;
            hostQ[i * stateSize + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    for (int i = 0; i < measurementSize; i++) {
        for (int j = 0; j < stateSize; j++) {
            hostH[i * stateSize + j] = (i == j) ? 1.0f : 0.0f;
        }
        hostR[i * measurementSize + i] = 0.1f;
    }

    // and now device memory
    float* deviceX = allocateDeviceMemory<float>(stateSize);
    float* deviceP = allocateDeviceMemory<float>(stateSize * stateSize);
    float* deviceF = allocateDeviceMemory<float>(stateSize * stateSize);
    float* deviceQ = allocateDeviceMemory<float>(stateSize * stateSize);
    float* deviceH = allocateDeviceMemory<float>(measurementSize * stateSize);
    float* deviceR = allocateDeviceMemory<float>(measurementSize * measurementSize);
    float* deviceZ = allocateDeviceMemory<float>(measurementSize);

    // copy data -> device
    copyToDevice(deviceX, hostX, stateSize);
    copyToDevice(deviceP, hostP, stateSize * stateSize);
    copyToDevice(deviceF, hostF, stateSize * stateSize);
    copyToDevice(deviceQ, hostQ, stateSize * stateSize);
    copyToDevice(deviceH, hostH, measurementSize * stateSize);
    copyToDevice(deviceR, hostR, measurementSize * measurementSize);
    
    // Set up
    dim3 block(256);
    dim3 grid((stateSize + block.x -1) / block.x);

    // Run for multiple time steps
    for (int i = 0; step < numSteps; step++) {
        // Random measurements
        for (int i = 0; i < measurementSize; i++) {
            hostZ[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        copyToDevice(deviceZ, hsotZ, measurementSize);

        // predict
        kalmanPrdict<<<grid, block>>>(deviceX, deviceP, deviceF, deviceQ, stateSize);
        cudeDeviceSynchronize();

        // update
        kalmanUpdate<<<grid, block>>>(deviceX, deviceP, deviceH, deviceR, deviceZ, stateSize, measurementSize);
        cudeDeviceSynchronize();
    }
}
