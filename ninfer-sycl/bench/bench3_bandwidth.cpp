// Week-0 microbench 3: raw memory bandwidth on the Arc.
//
// Goal: measure achievable device memory bandwidth with a simple read/write
// kernel. Expect ~450+ GB/s on the B770/B70 (GDDR6, ~512 GB/s peak). This is
// the ceiling that decode throughput is bound by.
//
// Three kernels:
//   - read-only  (each thread reads a strided chunk, reduces to a dummy sum)
//   - write-only (each thread writes a strided chunk)
//   - copy       (dst[i] = src[i])
// Bandwidth = bytes moved / time. For copy we count read+write.

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>

static double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    sycl::device dev{sycl::gpu_selector_v};
    sycl::queue q{dev};
    std::string name = dev.get_info<sycl::info::device::name>();
    printf("=== Raw memory bandwidth microbench (SYCL/Level Zero) ===\n");
    printf("Device: %s\n\n", name.c_str());

    const size_t N = 1u << 28;              // 268M elements
    const size_t BYTES = N * sizeof(float); // 1 GiB
    const int ITERS = 20;
    sycl::range<1> R(N);

    auto* a = sycl::malloc_device<float>(N, q);
    auto* b = sycl::malloc_device<float>(N, q);
    q.submit([&](sycl::handler& h) { h.fill(a, 1.0f, N); }).wait();

    // --- read-only ---
    {
        // warmup
        q.submit([&](sycl::handler& h) {
            h.parallel_for(R, [=](sycl::id<1> i) {
                float acc = a[i.get(0)];
                if (acc == 12345.0f) a[i.get(0)] = acc; // never true; keeps the read
            });
        }).wait();
        double t0 = now_ns();
        for (int it = 0; it < ITERS; it++) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(R, [=](sycl::id<1> i) {
                    float acc = a[i.get(0)];
                    if (acc == 12345.0f) a[i.get(0)] = acc;
                });
            });
        }
        q.wait();
        double t1 = now_ns();
        double ms = (t1 - t0) / ITERS / 1e6;
        printf("read-only : %8.3f ms  %7.1f GB/s\n", ms, BYTES / (ms * 1e-3) / 1e9);
    }

    // --- write-only ---
    {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(R, [=](sycl::id<1> i) { a[i.get(0)] = 2.0f; });
        }).wait();
        double t0 = now_ns();
        for (int it = 0; it < ITERS; it++) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(R, [=](sycl::id<1> i) { a[i.get(0)] = 2.0f; });
            });
        }
        q.wait();
        double t1 = now_ns();
        double ms = (t1 - t0) / ITERS / 1e6;
        printf("write-only: %8.3f ms  %7.1f GB/s\n", ms, BYTES / (ms * 1e-3) / 1e9);
    }

    // --- copy (count read+write) ---
    {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(R, [=](sycl::id<1> i) { b[i.get(0)] = a[i.get(0)]; });
        }).wait();
        double t0 = now_ns();
        for (int it = 0; it < ITERS; it++) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(R, [=](sycl::id<1> i) { b[i.get(0)] = a[i.get(0)]; });
            });
        }
        q.wait();
        double t1 = now_ns();
        double ms = (t1 - t0) / ITERS / 1e6;
        printf("copy      : %8.3f ms  %7.1f GB/s (r+w)\n", ms, 2.0 * BYTES / (ms * 1e-3) / 1e9);
    }

    sycl::free(a, q);
    sycl::free(b, q);
    printf("\nDone.\n");
    return 0;
}