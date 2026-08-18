// generate_fftw_wisdom.cpp
//
// Pre-generates FFTW wisdom for all FFT sizes required by PhaseEstimator and
// IAFEstimator, so that neither processor ever falls back to planning
// (FFTW_PATIENT/FFTW_ESTIMATE) at runtime.
//
// PhaseEstimator: covers f0 values from 5.0 Hz to 16.0 Hz in 0.1 Hz steps.
// The sizing logic mirrors PhaseEstimator exactly:
//   window_size = static_cast<int>(2.0 * fs / f0)
//   n_fft       = good_size_real(window_size)
// Two plan types are generated per unique n_fft, matching PhaseEstimator:
//   - fftw_plan_dft_r2c_1d  (forward, real-to-complex; matches p_)
//   - fftw_plan_dft_1d       (backward, complex-to-complex; matches p_inv_)
//
// IAFEstimator: a single fixed-size, r2c-forward-only plan (matches
// fft_plan_), sized from window_size_sec rather than f0:
//   window_size = window_size_sec * fs
//   n_fft       = good_size_real(window_size)
// This is a different sizing rule from PhaseEstimator's sweep above, so it
// is not automatically covered by it and is planned separately below.
//
// Build (both processors use double-precision FFTW):
//   g++ -O2 -o precompute_wisdom precompute_wisdom.cpp -lfftw3
//
// Usage:
//   ./precompute_wisdom
//   (wisdom file path, sample rate, and window_size_sec are hardcoded below
//   to match TurboLinkCLAS/ERPCLAS/ReplayCLAS.yaml; update them if those
//   graphs' fs/window_size_sec change)

#include <fftw3.h>
#include <cstdio>
#include <cmath>
#include <set>
#include <vector>

// ── USER CONFIGURATION ───────────────────────────────────────────────────────

// Full path to the FFTW wisdom file (loaded first, then updated in-place).
static const char* WISDOM_PATH = "resources/fft_wisdom/fftw_wisdom.txt";

// Sample rate used by PhaseEstimator (Hz).
static const double FS = 10000.0;

// f0 sweep parameters (Hz), for PhaseEstimator.
static const double F0_MIN  =  5.0;
static const double F0_MAX  = 16.0;
static const double F0_STEP =  0.1;

// window_size_sec used by IAFEstimator (see TurboLinkCLAS/ERPCLAS/ReplayCLAS.yaml).
static const double IAF_WINDOW_SIZE_SEC = 10.0;

// FFTW planner flag.  Use FFTW_MEASURE for thorough wisdom; FFTW_PATIENT is
// slower but produces even better plans.  FFTW_ESTIMATE skips measurement
// (fast but no real wisdom benefit).
static const unsigned PLANNER_FLAG = FFTW_MEASURE;

// ── HELPERS (identical to the anonymous-namespace helpers in PhaseEstimator) ─

static size_t good_size_real(size_t n)
{
    if (n <= 6)
        return n;

    size_t bestfac = 2 * n;
    for (size_t f5 = 1; f5 < bestfac; f5 *= 5)
    {
        size_t x = f5;
        while (x < n)
            x *= 2;
        for (;;)
        {
            if (x < n)
                x *= 3;
            else if (x > n)
            {
                if (x < bestfac)
                    bestfac = x;
                if (x & 1)
                    break;
                x >>= 1;
            }
            else
                return n;
        }
    }
    return bestfac;
}

// ── MAIN ─────────────────────────────────────────────────────────────────────

int main()
{
    // 1. Load existing wisdom so we don't discard plans already computed.
    int loaded = fftw_import_wisdom_from_filename(WISDOM_PATH);
    if (loaded)
        printf("[wisdom] Loaded existing wisdom from: %s\n", WISDOM_PATH);
    else
        printf("[wisdom] No existing wisdom file found at: %s — starting fresh.\n", WISDOM_PATH);

    // 2. Collect the unique n_fft values needed across the full f0 sweep.
    std::set<int> fft_sizes;
    {
        int steps = static_cast<int>(std::round((F0_MAX - F0_MIN) / F0_STEP)) + 1;
        for (int i = 0; i < steps; ++i)
        {
            double f0 = F0_MIN + i * F0_STEP;
            int window_size = static_cast<int>(2.0 * FS / f0);
            int n_fft       = static_cast<int>(good_size_real(static_cast<size_t>(window_size)));
            fft_sizes.insert(n_fft);
        }
    }

    printf("[wisdom] Sample rate : %.1f Hz\n", FS);
    printf("[wisdom] f0 range    : %.1f – %.1f Hz in %.1f Hz steps\n",
           F0_MIN, F0_MAX, F0_STEP);
    printf("[wisdom] Unique FFT sizes (%zu total):\n", fft_sizes.size());
    for (int s : fft_sizes)
        printf("         %d\n", s);
    printf("\n");

    // 3. Generate plans for each unique size.
    int idx = 0;
    for (int n : fft_sizes)
    {
        ++idx;
        printf("[%d/%d] Planning n_fft = %d ...\n",
               idx, static_cast<int>(fft_sizes.size()), n);

        // Allocate temporary buffers (FFTW requires aligned memory for planning).
        double*         real_buf  = fftw_alloc_real(n);
        fftw_complex* half_buf  = fftw_alloc_complex(n / 2 + 1);
        fftw_complex* cx_in     = fftw_alloc_complex(n);
        fftw_complex* cx_out    = fftw_alloc_complex(n);

        if (!real_buf || !half_buf || !cx_in || !cx_out)
        {
            fprintf(stderr, "  ERROR: allocation failed for n=%d\n", n);
            fftw_free(real_buf);
            fftw_free(half_buf);
            fftw_free(cx_in);
            fftw_free(cx_out);
            continue;
        }

        // Forward real-to-complex plan  (matches p_ in PhaseEstimator).
        fftw_plan p_fwd = fftw_plan_dft_r2c_1d(n, real_buf, half_buf, PLANNER_FLAG);
        if (p_fwd)
        {
            printf("  [r2c] plan created.\n");
            fftw_destroy_plan(p_fwd);
        }
        else
        {
            fprintf(stderr, "  [r2c] plan FAILED for n=%d\n", n);
        }

        // Backward complex-to-complex plan  (matches p_inv_ in PhaseEstimator).
        fftw_plan p_inv = fftw_plan_dft_1d(n, cx_in, cx_out, FFTW_BACKWARD, PLANNER_FLAG);
        if (p_inv)
        {
            printf("  [c2c backward] plan created.\n");
            fftw_destroy_plan(p_inv);
        }
        else
        {
            fprintf(stderr, "  [c2c backward] plan FAILED for n=%d\n", n);
        }

        fftw_free(real_buf);
        fftw_free(half_buf);
        fftw_free(cx_in);
        fftw_free(cx_out);
    }

    // 4. IAFEstimator's single fixed-size r2c-forward-only plan. Different
    //    sizing rule from PhaseEstimator (window_size_sec * fs, not f0-based),
    //    so it is not covered by the sweep above and must be planned here.
    {
        int n = static_cast<int>(good_size_real(static_cast<size_t>(IAF_WINDOW_SIZE_SEC * FS)));
        printf("[IAFEstimator] Planning n_fft = %d (window_size_sec=%.1f, fs=%.1f) ...\n",
               n, IAF_WINDOW_SIZE_SEC, FS);

        double*       real_buf = fftw_alloc_real(n);
        fftw_complex* half_buf = fftw_alloc_complex(n / 2 + 1);

        if (!real_buf || !half_buf)
        {
            fprintf(stderr, "  ERROR: allocation failed for n=%d\n", n);
        }
        else
        {
            // Forward real-to-complex plan (matches fft_plan_ in IAFEstimator;
            // IAFEstimator never plans a backward transform).
            fftw_plan p_fwd = fftw_plan_dft_r2c_1d(n, real_buf, half_buf, PLANNER_FLAG);
            if (p_fwd)
            {
                printf("  [r2c] plan created.\n");
                fftw_destroy_plan(p_fwd);
            }
            else
            {
                fprintf(stderr, "  [r2c] plan FAILED for n=%d\n", n);
            }
        }

        fftw_free(real_buf);
        fftw_free(half_buf);
    }

    // 5. Export accumulated wisdom back to the file.
    int saved = fftw_export_wisdom_to_filename(WISDOM_PATH);
    if (saved)
        printf("\n[wisdom] Wisdom saved successfully to: %s\n", WISDOM_PATH);
    else
        fprintf(stderr, "\n[wisdom] ERROR: failed to save wisdom to: %s\n", WISDOM_PATH);

    fftw_cleanup();
    return saved ? 0 : 1;
}