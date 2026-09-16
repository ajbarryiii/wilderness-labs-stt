// CPU-only packed inference. Baseline translation unit has no AVX requirement;
// target attributes and runtime CPUID checks guard the specialized entrypoints.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#define POPCNT __attribute__((target("popcnt")))
#define AVX512 __attribute__((target("avx512f,avx512vpopcntdq")))

namespace {
thread_local std::string last_error;

struct Weight {
    int64_t n, k, words, blocks;
    int weight_bits, activation_bits, backend, threads;
    // Scalar: [N, words]. AVX512: [ceil(N/8), words, 8].
    // Each SIMD lane accumulates an independent output channel.
    std::vector<uint64_t> sign, nonzero;
    std::vector<int64_t> counts;
    std::vector<float> scales;
    std::vector<uint64_t> activation_sign, activation_nonzero;

    size_t offset(int64_t row, int64_t word) const {
        return backend ? ((row / 8) * words + word) * 8 + row % 8
                       : row * words + word;
    }
};

bool has_popcnt() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("popcnt");
}
bool has_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512vpopcntdq");
}

template<bool A2>
void pack_scalar(const float* input, int64_t k, uint64_t* sign, uint64_t* nz) {
    for (int64_t start = 0; start < k; start += 64) {
        uint64_t s = 0, z = 0;
        const int64_t end = std::min(k - start, int64_t(64));
        for (int64_t bit = 0; bit < end; ++bit) {
            const float v = input[start + bit];
            if constexpr (A2) {
                s |= uint64_t(v >= 0.5f) << bit;
                z |= uint64_t(v >= 0.5f || v <= -0.5f) << bit;
            } else {
                s |= uint64_t(v >= 0.f) << bit;
            }
        }
        sign[start / 64] = s;
        if constexpr (A2) nz[start / 64] = z;
    }
}

template<bool A2>
AVX512 void pack_avx512(const float* input, int64_t k, uint64_t* sign, uint64_t* nz) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 positive = _mm512_set1_ps(0.5f);
    const __m512 negative = _mm512_set1_ps(-0.5f);
    for (int64_t start = 0; start < k; start += 64) {
        uint64_t s = 0, z = 0;
        for (int part = 0; part < 4 && start + 16 * part < k; ++part) {
            const int64_t i = start + 16 * part;
            const int valid = int(std::min(k - i, int64_t(16)));
            const __mmask16 mask = __mmask16((uint32_t(1) << valid) - 1);
            const __m512 v = _mm512_maskz_loadu_ps(mask, input + i);
            if constexpr (A2) {
                const __mmask16 pos = _mm512_cmp_ps_mask(v, positive, _CMP_GE_OQ) & mask;
                const __mmask16 neg = _mm512_cmp_ps_mask(v, negative, _CMP_LE_OQ) & mask;
                s |= uint64_t(pos) << (16 * part);
                z |= uint64_t(pos | neg) << (16 * part);
            } else {
                s |= uint64_t(_mm512_cmp_ps_mask(v, zero, _CMP_GE_OQ) & mask)
                     << (16 * part);
            }
        }
        sign[start / 64] = s;
        if constexpr (A2) nz[start / 64] = z;
    }
}

template<bool W2, bool A2>
POPCNT void dot_scalar(const Weight& w, const uint64_t* as, const uint64_t* az,
                      int64_t row, const float* bias, float* output) {
    const uint64_t* ws = w.sign.data() + row * w.words;
    const uint64_t* wz = W2 ? w.nonzero.data() + row * w.words : nullptr;
    int64_t mismatch = 0, count = 0;
    for (int64_t word = 0; word < w.words; ++word) {
        uint64_t active = W2 ? wz[word] : ~uint64_t(0);
        if constexpr (A2) active &= az[word];
        mismatch += __builtin_popcountll((ws[word] ^ as[word]) & active);
        if constexpr (A2) count += __builtin_popcountll(active);
    }
    if constexpr (!A2) count = W2 ? w.counts[row] : w.k;
    const float value = float(count - 2 * mismatch) * w.scales[row];
    output[row] = bias ? value + bias[row] : value;
}

template<bool W2, bool A2>
AVX512 void dot_avx512(const Weight& w, const uint64_t* as, const uint64_t* az,
                      int64_t block, const float* bias, float* output) {
    const uint64_t* ws = w.sign.data() + block * w.words * 8;
    const uint64_t* wz = W2 ? w.nonzero.data() + block * w.words * 8 : nullptr;
    __m512i mismatch0 = _mm512_setzero_si512(), mismatch1 = _mm512_setzero_si512();
    __m512i count0 = _mm512_setzero_si512(), count1 = _mm512_setzero_si512();
    for (int64_t word = 0; word < w.words; ++word) {
        const __m512i signs = _mm512_loadu_si512(ws + word * 8);
        const __m512i activation = _mm512_set1_epi64(as[word]);
        __m512i active = W2 ? _mm512_loadu_si512(wz + word * 8)
                           : _mm512_set1_epi64(-1);
        if constexpr (A2) active = _mm512_and_si512(active, _mm512_set1_epi64(az[word]));
        const __m512i different = _mm512_and_si512(_mm512_xor_si512(signs, activation), active);
        const __m512i mismatch = _mm512_popcnt_epi64(different);
        if (word & 1) mismatch1 = _mm512_add_epi64(mismatch1, mismatch);
        else mismatch0 = _mm512_add_epi64(mismatch0, mismatch);
        if constexpr (A2) {
            const __m512i count = _mm512_popcnt_epi64(active);
            if (word & 1) count1 = _mm512_add_epi64(count1, count);
            else count0 = _mm512_add_epi64(count0, count);
        }
    }
    const int64_t row = block * 8;
    __m512i count;
    if constexpr (A2) count = _mm512_add_epi64(count0, count1);
    else if constexpr (W2) count = _mm512_loadu_si512(w.counts.data() + row);
    else count = _mm512_set1_epi64(w.k);
    const __m512i mismatch = _mm512_add_epi64(mismatch0, mismatch1);
    const __m512i dot = _mm512_sub_epi64(count, _mm512_slli_epi64(mismatch, 1));
    alignas(64) int64_t values[8];
    _mm512_store_si512(values, dot);
    // Small scalar FP32 epilogue avoids introducing AVX512DQ as a requirement.
    for (int lane = 0; lane < 8 && row + lane < w.n; ++lane) {
        const float value = float(values[lane]) * w.scales[row + lane];
        output[row + lane] = bias ? value + bias[row + lane] : value;
    }
}

template<bool W2, bool A2, bool SIMD>
void linear(Weight& w, const float* x, int64_t m, const float* bias, float* out) {
    w.activation_sign.resize(m * w.words);
    if constexpr (A2) w.activation_nonzero.resize(m * w.words);
    uint64_t* as = w.activation_sign.data();
    uint64_t* az = A2 ? w.activation_nonzero.data() : nullptr;
    const int64_t units = SIMD ? w.blocks : w.n;
    const int thread_count = int(std::min(int64_t(w.threads), m * units));
    #pragma omp parallel num_threads(thread_count) if(thread_count > 1)
    {
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < m; ++row) {
            if constexpr (SIMD) pack_avx512<A2>(x + row * w.k, w.k,
                as + row * w.words, A2 ? az + row * w.words : nullptr);
            else pack_scalar<A2>(x + row * w.k, w.k,
                as + row * w.words, A2 ? az + row * w.words : nullptr);
        }
        #pragma omp for schedule(static)
        for (int64_t task = 0; task < m * units; ++task) {
            const int64_t row = task / units;
            const int64_t channel = task % units;
            if constexpr (SIMD) dot_avx512<W2, A2>(w, as + row * w.words,
                A2 ? az + row * w.words : nullptr, channel, bias, out + row * w.n);
            else dot_scalar<W2, A2>(w, as + row * w.words,
                A2 ? az + row * w.words : nullptr, channel, bias, out + row * w.n);
        }
    }
}

template<bool SIMD>
void dispatch(Weight& w, const float* x, int64_t m, const float* bias, float* out) {
    if (w.weight_bits == 1 && w.activation_bits == 1) linear<false,false,SIMD>(w,x,m,bias,out);
    else if (w.weight_bits == 2 && w.activation_bits == 1) linear<true,false,SIMD>(w,x,m,bias,out);
    else if (w.weight_bits == 1) linear<false,true,SIMD>(w,x,m,bias,out);
    else linear<true,true,SIMD>(w,x,m,bias,out);
}

// The original scalar and AVX512 routines above remain available as baselines.
// Optimized kernels retain their packed layout and exact arithmetic, but reuse
// every weight vector across several input rows and vectorize the FP32 epilogue.
AVX512 void store_optimized(const Weight& w, __m512i dot, int64_t channel,
                           const float* bias, float* output) {
    if (channel + 8 <= w.n) {
        // The mathematical dot is in [-K,K], and construction limits K to
        // INT32_MAX. Narrowing the completed int64 sum is therefore exact and
        // avoids requiring AVX512DQ's int64-to-float conversion instruction.
        const __m256 values = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(dot));
        const __m256 scaled = _mm256_mul_ps(values, _mm256_loadu_ps(w.scales.data() + channel));
        const __m256 result = bias ? _mm256_add_ps(scaled, _mm256_loadu_ps(bias + channel))
                                   : scaled;
        _mm256_storeu_ps(output + channel, result);
    } else {
        alignas(64) int64_t values[8];
        _mm512_store_si512(values, dot);
        for (int lane = 0; channel + lane < w.n; ++lane) {
            const float value = float(values[lane]) * w.scales[channel + lane];
            output[channel + lane] = bias ? value + bias[channel + lane] : value;
        }
    }
}

template<bool W2, bool A2, int Rows>
AVX512 void dot_tile(const Weight& w, const uint64_t* as, const uint64_t* az,
                    int64_t block, const float* bias, float* output) {
    const uint64_t* ws = w.sign.data() + block * w.words * 8;
    const uint64_t* wz = W2 ? w.nonzero.data() + block * w.words * 8 : nullptr;
    __m512i mismatch[Rows], count[Rows];
    #pragma GCC unroll 8
    for (int row = 0; row < Rows; ++row) {
        mismatch[row] = _mm512_setzero_si512();
        if constexpr (A2) count[row] = _mm512_setzero_si512();
    }
    for (int64_t word = 0; word < w.words; ++word) {
        const __m512i signs = _mm512_loadu_si512(ws + word * 8);
        const __m512i weight_active = W2 ? _mm512_loadu_si512(wz + word * 8)
                                        : _mm512_set1_epi64(-1);
        #pragma GCC unroll 8
        for (int row = 0; row < Rows; ++row) {
            const __m512i activation = _mm512_set1_epi64(as[row * w.words + word]);
            __m512i active = weight_active;
            if constexpr (A2)
                active = _mm512_and_si512(active, _mm512_set1_epi64(az[row * w.words + word]));
            // Keep the shared active mask available for its population count;
            // explicitly fuse the subsequent XOR and mask into one operation.
            const __m512i different = (W2 || A2)
                ? _mm512_ternarylogic_epi64(signs, activation, active, 0x28)
                : _mm512_xor_si512(signs, activation);
            mismatch[row] = _mm512_add_epi64(mismatch[row], _mm512_popcnt_epi64(different));
            if constexpr (A2)
                count[row] = _mm512_add_epi64(count[row], _mm512_popcnt_epi64(active));
        }
    }
    const int64_t channel = block * 8;
    if (channel + 8 <= w.n) {
        const __m256 scale = _mm256_loadu_ps(w.scales.data() + channel);
        const __m256 row_bias = bias ? _mm256_loadu_ps(bias + channel) : _mm256_setzero_ps();
        __m512i shared_count;
        if constexpr (!A2 && W2) shared_count = _mm512_loadu_si512(w.counts.data() + channel);
        else if constexpr (!A2) shared_count = _mm512_set1_epi64(w.k);
        #pragma GCC unroll 8
        for (int row = 0; row < Rows; ++row) {
            const __m512i active_count = A2 ? count[row] : shared_count;
            const __m512i dot = _mm512_sub_epi64(active_count, _mm512_slli_epi64(mismatch[row], 1));
            const __m256 values = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(dot));
            const __m256 scaled = _mm256_mul_ps(values, scale);
            _mm256_storeu_ps(output + row * w.n + channel,
                            bias ? _mm256_add_ps(scaled, row_bias) : scaled);
        }
        return;
    }
    #pragma GCC unroll 8
    for (int row = 0; row < Rows; ++row) {
        __m512i active_count;
        if constexpr (A2) active_count = count[row];
        else if constexpr (W2) active_count = _mm512_loadu_si512(w.counts.data() + channel);
        else active_count = _mm512_set1_epi64(w.k);
        const __m512i dot = _mm512_sub_epi64(active_count, _mm512_slli_epi64(mismatch[row], 1));
        store_optimized(w, dot, channel, bias, output + row * w.n);
    }
}

template<bool W2, bool A2>
AVX512 void dot_single_optimized(const Weight& w, const uint64_t* as, const uint64_t* az,
                                int64_t block, const float* bias, float* output) {
    const uint64_t* ws = w.sign.data() + block * w.words * 8;
    const uint64_t* wz = W2 ? w.nonzero.data() + block * w.words * 8 : nullptr;
    __m512i mismatch0 = _mm512_setzero_si512(), mismatch1 = _mm512_setzero_si512();
    __m512i count0 = _mm512_setzero_si512(), count1 = _mm512_setzero_si512();
    int64_t word = 0;
    for (; word + 1 < w.words; word += 2) {
        const __m512i signs0 = _mm512_loadu_si512(ws + word * 8);
        const __m512i signs1 = _mm512_loadu_si512(ws + (word + 1) * 8);
        __m512i active0 = W2 ? _mm512_loadu_si512(wz + word * 8) : _mm512_set1_epi64(-1);
        __m512i active1 = W2 ? _mm512_loadu_si512(wz + (word + 1) * 8) : _mm512_set1_epi64(-1);
        if constexpr (A2) {
            active0 = _mm512_and_si512(active0, _mm512_set1_epi64(az[word]));
            active1 = _mm512_and_si512(active1, _mm512_set1_epi64(az[word + 1]));
            count0 = _mm512_add_epi64(count0, _mm512_popcnt_epi64(active0));
            count1 = _mm512_add_epi64(count1, _mm512_popcnt_epi64(active1));
        }
        const __m512i different0 = _mm512_and_si512(_mm512_xor_si512(signs0, _mm512_set1_epi64(as[word])), active0);
        const __m512i different1 = _mm512_and_si512(_mm512_xor_si512(signs1, _mm512_set1_epi64(as[word + 1])), active1);
        mismatch0 = _mm512_add_epi64(mismatch0, _mm512_popcnt_epi64(different0));
        mismatch1 = _mm512_add_epi64(mismatch1, _mm512_popcnt_epi64(different1));
    }
    if (word < w.words) {
        const __m512i signs = _mm512_loadu_si512(ws + word * 8);
        __m512i active = W2 ? _mm512_loadu_si512(wz + word * 8) : _mm512_set1_epi64(-1);
        if constexpr (A2) {
            active = _mm512_and_si512(active, _mm512_set1_epi64(az[word]));
            count0 = _mm512_add_epi64(count0, _mm512_popcnt_epi64(active));
        }
        const __m512i different = _mm512_and_si512(_mm512_xor_si512(signs, _mm512_set1_epi64(as[word])), active);
        mismatch0 = _mm512_add_epi64(mismatch0, _mm512_popcnt_epi64(different));
    }
    const int64_t channel = block * 8;
    __m512i count;
    if constexpr (A2) count = _mm512_add_epi64(count0, count1);
    else if constexpr (W2) count = _mm512_loadu_si512(w.counts.data() + channel);
    else count = _mm512_set1_epi64(w.k);
    const __m512i mismatch = _mm512_add_epi64(mismatch0, mismatch1);
    store_optimized(w, _mm512_sub_epi64(count, _mm512_slli_epi64(mismatch, 1)),
                    channel, bias, output);
}

template<bool W2, bool A2>
void linear_optimized(Weight& w, const float* x, int64_t m, const float* bias, float* out) {
    w.activation_sign.resize(m * w.words);
    if constexpr (A2) w.activation_nonzero.resize(m * w.words);
    uint64_t* as = w.activation_sign.data();
    uint64_t* az = A2 ? w.activation_nonzero.data() : nullptr;
    if (m == 1) {
        // Packing a single row is serial either way. Do it before entering the
        // worker team, eliminating the otherwise unnecessary packing barrier.
        pack_avx512<A2>(x, w.k, as, az);
        // On this CPU the measured team startup cost exceeds the benefit for
        // small GEMVs: up to 384 KiB of binary planes or 256 KiB of ternary
        // planes. Larger projections, including vocabulary, retain the team.
        constexpr int64_t serial_words = W2 ? 16384 : 49152;
        const bool small_projection = w.n <= serial_words / w.words;
        const int thread_count = small_projection ? 1
            : int(std::min(int64_t(w.threads), w.blocks));
        #pragma omp parallel for schedule(static) num_threads(thread_count) if(thread_count > 1)
        for (int64_t block = 0; block < w.blocks; ++block)
            dot_single_optimized<W2, A2>(w, as, az, block, bias, out);
        return;
    }
    constexpr int tile_rows = 8;
    const int64_t tiles = (m + tile_rows - 1) / tile_rows;
    const int thread_count = int(std::min(int64_t(w.threads), tiles * w.blocks));
    #pragma omp parallel num_threads(thread_count) if(thread_count > 1)
    {
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < m; ++row)
            pack_avx512<A2>(x + row * w.k, w.k, as + row * w.words,
                           A2 ? az + row * w.words : nullptr);
        #pragma omp for schedule(static)
        for (int64_t task = 0; task < tiles * w.blocks; ++task) {
            const int64_t row = (task / w.blocks) * tile_rows;
            const int64_t block = task % w.blocks;
            const uint64_t* row_sign = as + row * w.words;
            const uint64_t* row_nz = A2 ? az + row * w.words : nullptr;
            float* row_output = out + row * w.n;
            int remaining = int(std::min(int64_t(tile_rows), m - row));
            if (remaining == 8) {
                dot_tile<W2,A2,8>(w,row_sign,row_nz,block,bias,row_output);
                continue;
            }
            if (remaining >= 4) {
                dot_tile<W2,A2,4>(w,row_sign,row_nz,block,bias,row_output);
                remaining -= 4;
                row_sign += 4 * w.words;
                if constexpr (A2) row_nz += 4 * w.words;
                row_output += 4 * w.n;
            }
            switch (remaining) {
                case 3: dot_tile<W2,A2,3>(w,row_sign,row_nz,block,bias,row_output); break;
                case 2: dot_tile<W2,A2,2>(w,row_sign,row_nz,block,bias,row_output); break;
                case 1: dot_single_optimized<W2,A2>(w,row_sign,row_nz,block,bias,row_output); break;
            }
        }
    }
}

void dispatch_optimized(Weight& w, const float* x, int64_t m, const float* bias, float* out) {
    if (w.weight_bits == 1 && w.activation_bits == 1) linear_optimized<false,false>(w,x,m,bias,out);
    else if (w.weight_bits == 2 && w.activation_bits == 1) linear_optimized<true,false>(w,x,m,bias,out);
    else if (w.weight_bits == 1) linear_optimized<false,true>(w,x,m,bias,out);
    else linear_optimized<true,true>(w,x,m,bias,out);
}
} // namespace

extern "C" {
const char* cpu_last_error() { return last_error.c_str(); }
int cpu_feature_flags() { return (has_popcnt() ? 1 : 0) | (has_avx512() ? 2 : 0); }

void* cpu_weight_create(const int8_t* codes, const float* scales, int64_t n, int64_t k,
                        int weight_bits, int activation_bits, int backend, int threads) {
    try {
        if (n < 1 || k < 1 || k > std::numeric_limits<int32_t>::max() || threads < 1 ||
            (weight_bits != 1 && weight_bits != 2) ||
            (activation_bits != 1 && activation_bits != 2) ||
            (backend != 0 && backend != 1 && backend != 2)) throw std::invalid_argument("invalid weight configuration");
        if (!has_popcnt() || (backend && !has_avx512()))
            throw std::runtime_error("requested CPU instruction set is unavailable");
        auto w = std::make_unique<Weight>();
        w->n=n; w->k=k; w->words=(k+63)/64; w->blocks=(n+7)/8;
        w->weight_bits=weight_bits; w->activation_bits=activation_bits;
        w->backend=backend; w->threads=threads;
        const int64_t padded_n = backend ? w->blocks * 8 : n;
        w->sign.resize(padded_n * w->words, 0);
        if (weight_bits == 2) {
            w->nonzero.resize(padded_n * w->words, 0);
            w->counts.resize(padded_n, 0);
        }
        w->scales.assign(scales, scales+n);
        for (int64_t row = 0; row < n; ++row) {
            int64_t count = 0;
            for (int64_t col = 0; col < k; ++col) {
                const int code = codes[row * k + col];
                if (code < -1 || code > 1 || (weight_bits == 1 && code == 0))
                    throw std::invalid_argument("weight codes do not match the requested precision");
                const size_t at = w->offset(row, col / 64);
                const uint64_t bit = uint64_t(1) << (col % 64);
                if (code > 0) w->sign[at] |= bit;
                if (weight_bits == 2 && code != 0) { w->nonzero[at] |= bit; ++count; }
            }
            if (weight_bits == 2) w->counts[row] = count;
        }
        return w.release();
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}

void cpu_weight_destroy(void* handle) { delete static_cast<Weight*>(handle); }
uint64_t cpu_weight_storage_bytes(const void* handle) {
    const Weight& w = *static_cast<const Weight*>(handle);
    return (w.sign.size()+w.nonzero.size())*8 + w.counts.size()*8 + w.scales.size()*4;
}
uint64_t cpu_weight_scratch_bytes(const void* handle) {
    const Weight& w = *static_cast<const Weight*>(handle);
    return (w.activation_sign.capacity()+w.activation_nonzero.capacity())*8;
}

int cpu_weight_linear(void* handle, const float* x, int64_t m, const float* bias, float* out) {
    try {
        if (m < 0) throw std::invalid_argument("negative input row count");
        if (m == 0) return 0;
        Weight& w = *static_cast<Weight*>(handle);
        if (w.backend == 2) dispatch_optimized(w,x,m,bias,out);
        else if (w.backend) dispatch<true>(w,x,m,bias,out);
        else dispatch<false>(w,x,m,bias,out);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}

int cpu_weight_embedding(const void* handle, const int64_t* ids, int64_t m, float* out) {
    try {
        const Weight& w = *static_cast<const Weight*>(handle);
        for (int64_t i = 0; i < m; ++i) {
            const int64_t row = ids[i];
            if (row < 0 || row >= w.n) throw std::invalid_argument("embedding index out of range");
            for (int64_t col = 0; col < w.k; ++col) {
                const size_t at = w.offset(row, col/64);
                const uint64_t bit = uint64_t(1) << (col%64);
                const int code = (w.weight_bits == 2 && !(w.nonzero[at]&bit)) ? 0
                               : ((w.sign[at]&bit) ? 1 : -1);
                out[i*w.k+col] = float(code) * w.scales[row];
            }
        }
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
}
}
