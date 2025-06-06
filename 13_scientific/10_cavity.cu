#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm> 

#define BLOCK_SIZE 16

__global__ void compute_b_matrix(float *b, float *u, float *v, float dx, float dy, float dt, float rho, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        b[j * nx + i] = rho * (1.0f / dt *
                         ((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx) +
                          (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy)) -
                         powf((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx), 2) -
                         2.0f * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0f * dy) *
                                 (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2.0f * dx)) -
                         powf((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy), 2));
    }
}

__global__ void compute_p_matrix(float *p, float *pn, float *b, float dx, float dy, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        p[j * nx + i] = dy * dy * ((pn[j * nx + (i + 1)] + pn[j * nx + (i - 1)]) +
                        dx * dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                        b[j * nx + i] * dx * dx * dy * dy) /
                        (2.0f * (dx * dx + dy * dy));
    }
}

__global__ void update_uv_matrix(float *u, float *v, float *un, float *vn, float *p,
                                 float dx, float dy, float dt, float rho, float nu, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + (i - 1)]) -
                        vn[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                        dt / (2.0f * rho * dx) * (p[j * nx + (i + 1)] - p[j * nx + (i - 1)]) +
                        nu * dt / (dx * dx) * (un[j * nx + (i + 1)] - 2.0f * un[j * nx + i] + un[j * nx + (i - 1)]) +
                        nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2.0f * un[j * nx + i] + un[(j - 1) * nx + i]);

        v[j * nx + i] = vn[j * nx + i] - un[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + (i - 1)]) -
                        vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                        dt / (2.0f * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                        nu * dt / (dx * dx) * (vn[j * nx + (i + 1)] - 2.0f * vn[j * nx + i] + vn[j * nx + (i - 1)]) +
                        nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2.0f * vn[j * nx + i] + vn[(j - 1) * nx + i]);
    }
}

__global__ void set_p_boundaries(float* p, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        p[idx * nx + 0] = p[idx * nx + 1];
        p[idx * nx + (nx - 1)] = p[idx * nx + (nx - 2)];
    }
    if (idx < nx) {
        p[0 * nx + idx] = p[1 * nx + idx];
        p[(ny - 1) * nx + idx] = 0.0f;
    }
}

__global__ void set_uv_boundaries(float* u, float* v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        u[idx * nx + 0] = 0.0f;
        u[idx * nx + (nx - 1)] = 0.0f;
        v[idx * nx + 0] = 0.0f;
        v[idx * nx + (nx - 1)] = 0.0f;
    }
    if (idx < nx) {
        u[0 * nx + idx] = 0.0f;
        u[(ny - 1) * nx + idx] = 1.0f;
        v[0 * nx + idx] = 0.0f;
        v[(ny - 1) * nx + idx] = 0.0f;
    }
}

int main() {
    const int nx = 41;
    const int ny = 41;
    const int nt = 500;
    const int nit = 50;
    int size = nx * ny * sizeof(float);

    const float dx = 2.0f / (nx - 1);
    const float dy = 2.0f / (ny - 1);
    const float dt = 0.01f;
    const float rho = 1.0f;
    const float nu = 0.02f;

    float *u, *v, *p, *b, *un, *vn, *pn;
    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&p, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&un, size);
    cudaMallocManaged(&vn, size);
    cudaMallocManaged(&pn, size);

    for (int i = 0; i < nx * ny; ++i) {
        u[i] = v[i] = p[i] = b[i] = 0.0f;
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    std::ofstream ufile("uCUDA.dat"), vfile("vCUDA.dat"), pfile("pCUDA.dat");

    int maxDim = std::max(nx, ny);
    int threads = 256;
    int blocks = (maxDim + threads - 1) / threads;

    for (int n = 0; n < nt; n++) {
        compute_b_matrix<<<grid, block>>>(b, u, v, dx, dy, dt, rho, nx, ny);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++) {
            cudaMemcpy(pn, p, size, cudaMemcpyDeviceToDevice);
            compute_p_matrix<<<grid, block>>>(p, pn, b, dx, dy, nx, ny);
            cudaDeviceSynchronize();
            set_p_boundaries<<<blocks, threads>>>(p, nx, ny);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(un, u, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vn, v, size, cudaMemcpyDeviceToDevice);

        update_uv_matrix<<<grid, block>>>(u, v, un, vn, p, dx, dy, dt, rho, nu, nx, ny);
        cudaDeviceSynchronize();
        set_uv_boundaries<<<blocks, threads>>>(u, v, nx, ny);
        cudaDeviceSynchronize();

        if (n % 10 == 0) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) ufile << u[j * nx + i] << " ";
                ufile << "\n";
            }
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) vfile << v[j * nx + i] << " ";
                vfile << "\n";
            }
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) pfile << p[j * nx + i] << " ";
                pfile << "\n";
            }
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);

    ufile.close();
    vfile.close();
    pfile.close();

    return 0;
}

