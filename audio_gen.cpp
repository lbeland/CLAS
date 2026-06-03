/*
 * pink_burst.cpp
 *
 * Continuously streams silence via ALSA. Press ENTER to trigger a
 * 20 ms pink noise burst at any time. The audio thread picks up the
 * trigger at the next period boundary and plays the burst once.
 *
 * BUILD:
 *   g++ -O2 -o pink_burst pink_burst.cpp -lasound -lpthread
 *
 * RUN:
 *   ./pink_burst
 *   Press ENTER to trigger a burst. Ctrl+C to quit.
 */

#include <alsa/asoundlib.h>
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <vector>

// ── Parameters ────────────────────────────────────────────────────────────────
static const int   SAMPLE_RATE   = 44100;
static const int   CHANNELS      = 2;
static const int   BURST_MS      = 20;
static const int   PERIOD_MS     = 5;    // audio thread wakeup interval (ms)
static const float AMPLITUDE     = 0.7f;
static const int   NUM_OCTAVES   = 6;

static const int BURST_FRAMES  = BURST_MS  * SAMPLE_RATE / 1000;  //  882
static const int PERIOD_FRAMES = PERIOD_MS * SAMPLE_RATE / 1000;  //  220

// ── Shared state ──────────────────────────────────────────────────────────────
static std::atomic<bool> trigger_pending {false};  // main  → audio thread
static std::atomic<bool> running         {true};

// ── Precomputed buffers ───────────────────────────────────────────────────────
static std::vector<float> burst_buf;    // BURST_FRAMES  * 2 floats
static std::vector<float> silence_buf;  // PERIOD_FRAMES * 2 floats (all zero)

static void build_buffers() {
    // Pink noise (Voss-McCartney)
    float bands[NUM_OCTAVES] = {};
    unsigned int counter = 0;
    srand(42);
    auto white = [] { return (float)rand() / RAND_MAX * 2.f - 1.f; };

    std::vector<float> mono(BURST_FRAMES);
    for (int i = 0; i < BURST_FRAMES; ++i) {
        ++counter;
        for (int b = 0; b < NUM_OCTAVES; ++b)
            if ((counter >> b) & 1) bands[b] = white();
        float sum = 0;
        for (float v : bands) sum += v;
        mono[i] = sum / NUM_OCTAVES;
    }

    float peak = 0;
    for (float v : mono) peak = std::max(peak, std::abs(v));
    for (float& v : mono) v = v / peak * AMPLITUDE;

    burst_buf.resize(BURST_FRAMES * CHANNELS);
    for (int i = 0; i < BURST_FRAMES; ++i) {
        burst_buf[i * 2    ] = mono[i];
        burst_buf[i * 2 + 1] = mono[i];
    }

    silence_buf.assign(PERIOD_FRAMES * CHANNELS, 0.0f);
}

// ── ALSA setup ────────────────────────────────────────────────────────────────
static snd_pcm_t* open_alsa() {
    snd_pcm_t* pcm;
    if (snd_pcm_open(&pcm, "hw:1,0", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
        fprintf(stderr, "Cannot open audio device\n");
        exit(1);
    }

    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm, hw);
    snd_pcm_hw_params_set_access  (pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format  (pcm, hw, SND_PCM_FORMAT_FLOAT_LE);
    snd_pcm_hw_params_set_channels(pcm, hw, CHANNELS);
    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, 0);

    snd_pcm_uframes_t period = PERIOD_FRAMES;
    snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, 0);
    snd_pcm_uframes_t bufsize = period * 8;   // enough headroom
    snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &bufsize);

    snd_pcm_hw_params(pcm, hw);
    snd_pcm_prepare(pcm);
    return pcm;
}

// ── Audio thread: runs forever, plays burst or silence each period ─────────────
static void* audio_thread(void*) {
    snd_pcm_t* pcm = open_alsa();

    while (running) {
        if (trigger_pending.exchange(false)) {
            // Play the full burst in one write (~20 ms)
            int err = snd_pcm_writei(pcm, burst_buf.data(), BURST_FRAMES);
            if (err == -EPIPE) {
                snd_pcm_prepare(pcm);
                snd_pcm_writei(pcm, burst_buf.data(), BURST_FRAMES);
            }
        } else {
            // Play one period of silence (~5 ms)
            int err = snd_pcm_writei(pcm, silence_buf.data(), PERIOD_FRAMES);
            if (err == -EPIPE) {
                snd_pcm_prepare(pcm);
                snd_pcm_writei(pcm, silence_buf.data(), PERIOD_FRAMES);
            }
        }
    }

    snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    return nullptr;
}

// ── Signal handler ────────────────────────────────────────────────────────────
static void on_sigint(int) { running = false; }

// ── Main: trigger on ENTER (replace this with your real trigger source) ───────
int main() {
    signal(SIGINT, on_sigint);
    build_buffers();

    pthread_t tid;
    pthread_create(&tid, nullptr, audio_thread, nullptr);

    printf("Ready. Press ENTER to trigger a burst. Ctrl+C to quit.\n");

    while (running) {
        int c = getchar();
        if (c == '\n')
            trigger_pending.store(true);
    }

    pthread_join(tid, nullptr);
    printf("Done.\n");
    return 0;
}