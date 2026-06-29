/*
 * audio_gen.cpp
 *
 * Continuously streams silence via ALSA. Press ENTER to trigger a
 * 20 ms pink noise burst at any time. The audio thread picks up the
 * trigger at the next period boundary and plays the burst once.
 *
 * BUILD:
 *   g++ -O2 -o audio_gen audio_gen.cpp -lasound -lpthread
 *
 * RUN:
 *   ./audio_gen
 *   Press ENTER to trigger a burst. Ctrl+C to quit.
 */

#include <alsa/asoundlib.h>
#include "lib/miniaudio/miniaudio.c"
#include <alsa/asoundlib.h>
#include <stdio.h>
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <vector>

// ── Parameters ────────────────────────────────────────────────────────────────
static const int   SAMPLE_RATE   = 44100;
static const int   CHANNELS      = 2;
static const float   BURST_MS      = 1000.0;
static const float   PERIOD_MS     = 5;    // audio thread wakeup interval (ms)
static const float AMPLITUDE     = 0.7f;
static const int   NUM_OCTAVES   = 6;

static const int BURST_FRAMES  = (int)(BURST_MS  * SAMPLE_RATE / 1000.0);  //  882
static const int PERIOD_FRAMES = (int)(PERIOD_MS * SAMPLE_RATE / 1000);  //  220

// ── Shared state ──────────────────────────────────────────────────────────────
static std::atomic<bool> trigger_pending {false};  // main  → audio thread
static std::atomic<bool> running         {true};

// ── Precomputed buffers ───────────────────────────────────────────────────────
static std::vector<float> burst_buf;    // BURST_FRAMES  * 2 floats
static std::vector<float> silence_buf;  // PERIOD_FRAMES * 2 floats (all zero)

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
    if (pDecoder == NULL) {
        return;
    }

    ma_data_source_read_pcm_frames(pDecoder, pOutput, frameCount, NULL);

    (void)pInput;
}

static void build_buffers() {
    // Pink noise (Voss-McCartney)
    // float bands[NUM_OCTAVES] = {};
    // unsigned int counter = 0;
    // srand(42);
    // auto white = [] { return (float)rand() / RAND_MAX * 2.f - 1.f; };

    // std::vector<float> mono(BURST_FRAMES);
    // for (int i = 0; i < BURST_FRAMES; ++i) {
    //     ++counter;
    //     for (int b = 0; b < NUM_OCTAVES; ++b)
    //         if ((counter >> b) & 1) bands[b] = white();
    //     float sum = 0;
    //     for (float v : bands) sum += v;
    //     mono[i] = sum / NUM_OCTAVES;
    // }

    ma_result result;
    ma_decoder decoder;

    const char* file = "resources/sounds/background.mp3";

    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, 44100);
    result = ma_decoder_init_file(file, &config, &decoder);
    if (result != MA_SUCCESS) {
        printf("Failed to initialize decoder for %s\n", file);
    }
    else {
        printf("Successfully initialized decoder for %s\n", file);
    }

    std::vector<float> mono(BURST_FRAMES);
    result = ma_decoder_read_pcm_frames(&decoder, mono.data(), BURST_FRAMES, NULL);
    if (result != MA_SUCCESS) {
        printf("Failed to read PCM frames for %s\n", file);
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
    if (snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
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

// // ── Main: trigger on ENTER (replace this with your real trigger source) ───────
// int main() {

    
//     signal(SIGINT, on_sigint);
//     build_buffers();

//     pthread_t tid;
//     pthread_create(&tid, nullptr, audio_thread, nullptr);

//     printf("Ready. Press ENTER to trigger a burst. Ctrl+C to quit.\n");

//     while (running) {
//         int c = getchar();
//         if (c == '\n')
//             trigger_pending.store(true);
//     }

//     pthread_join(tid, nullptr);
//     printf("Done.\n");
//     return 0;
// }

int main(int argc, char** argv)
{
    ma_result result;
    ma_decoder decoder;
    ma_device_config deviceConfig;
    ma_device device;

    if (argc < 2) {
        printf("No input file.\n");
        return -1;
    }

    result = ma_decoder_init_file(argv[1], NULL, &decoder);
    if (result != MA_SUCCESS) {
        return -2;
    }
    ma_data_source_set_looping(&decoder, MA_TRUE);

    deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = decoder.outputFormat;
    deviceConfig.playback.channels = decoder.outputChannels;
    deviceConfig.sampleRate        = decoder.outputSampleRate;
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &decoder;

    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        printf("Failed to open playback device.\n");
        ma_decoder_uninit(&decoder);
        return -3;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -4;
    }

    printf("Press Enter to quit...");
    getchar();
    ma_device_stop(&device);

    printf("Press Enter to continue...");
    getchar();
    if (ma_device_start(&device) != MA_SUCCESS) {
        printf("Failed to start playback device.\n");
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        return -4;
    }
    printf("Press Enter to quit...");
    getchar();

    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);

    return 0;
}