// src/main.c
#include <string.h>
#include <zephyr/drivers/i2s.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/atomic.h>

/* ==== Audio format ==== */
#define SAMPLE_RATE 48000
#define SAMPLE_BIT_WIDTH 16
#define NUM_CHANNELS 2
#define BYTES_PER_SAMPLE (SAMPLE_BIT_WIDTH / 8)

/* ~21.3 ms per block at 48 kHz (comparable to 256 @ 11025) */
#define SAMPLES_PER_BLOCK 1024 /* frames per channel */
#define SAMPLES_PER_BUFFER (SAMPLES_PER_BLOCK * NUM_CHANNELS)
#define BUFFER_SIZE_BYTES (SAMPLES_PER_BUFFER * BYTES_PER_SAMPLE)
BUILD_ASSERT(BYTES_PER_SAMPLE == 2, "Only 16-bit samples are used here.");

/* Device */
#define I2S_DEV DT_ALIAS(i2s_tx)

/* Driver TX slab (internal queue). App does NOT allocate from this when using
 * i2s_buf_write(). */
K_MEM_SLAB_DEFINE(tx_mem_slab, BUFFER_SIZE_BYTES, 8, 4);

/* App queue: main() -> feeder; we pass pointers to raw buffers to be copied by
 * i2s_buf_write(). */
#define QUEUE_DEPTH 4
K_MSGQ_DEFINE(audio_q, sizeof(void *), QUEUE_DEPTH, 4);

/* ==== Tone generator (LUT) ==== */
#define SINE_TABLE_SIZE 256
static const int16_t sine_lut[SINE_TABLE_SIZE] = {
    0,      804,    1608,   2410,   3212,   4011,   4808,   5602,   6393,
    7179,   7962,   8739,   9512,   10278,  11039,  11794,  12542,  13283,
    14017,  14744,  15464,  16175,  16878,  17572,  18257,  18933,  19599,
    20256,  20902,  21538,  22164,  22778,  23381,  23972,  24551,  25117,
    25671,  26211,  26739,  27253,  27753,  28239,  28711,  29168,  29611,
    30039,  30451,  30849,  31231,  31598,  31950,  32286,  32606,  32911,
    33199,  33472,  33728,  33968,  34192,  34400,  34591,  34766,  34925,
    35067,  35192,  35301,  35393,  35468,  35526,  35568,  35593,  35601,
    35593,  35568,  35526,  35468,  35393,  35301,  35192,  35067,  34925,
    34766,  34591,  34400,  34192,  33968,  33728,  33472,  33199,  32911,
    32606,  32286,  31950,  31598,  31231,  30849,  30451,  30039,  29611,
    29168,  28711,  28239,  27753,  27253,  26739,  26211,  25671,  25117,
    24551,  23972,  23381,  22778,  22164,  21538,  20902,  20256,  19599,
    18933,  18257,  17572,  16878,  16175,  15464,  14744,  14017,  13283,
    12542,  11794,  11039,  10278,  9512,   8739,   7962,   7179,   6393,
    5602,   4808,   4011,   3212,   2410,   1608,   804,    0,      -804,
    -1608,  -2410,  -3212,  -4011,  -4808,  -5602,  -6393,  -7179,  -7962,
    -8739,  -9512,  -10278, -11039, -11794, -12542, -13283, -14017, -14744,
    -15464, -16175, -16878, -17572, -18257, -18933, -19599, -20256, -20902,
    -21538, -22164, -22778, -23381, -23972, -24551, -25117, -25671, -26211,
    -26739, -27253, -27753, -28239, -28711, -29168, -29611, -30039, -30451,
    -30849, -31231, -31598, -31950, -32286, -32606, -32911, -33199, -33472,
    -33728, -33968, -34192, -34400, -34591, -34766, -34925, -35067, -35192,
    -35301, -35393, -35468, -35526, -35568, -35593, -35601, -35593, -35568,
    -35526};
#define TONE_HZ 440.0f
static uint32_t phase_acc;
static const uint32_t phase_step =
    (uint32_t)((TONE_HZ / SAMPLE_RATE) * SINE_TABLE_SIZE * 65536);

/* Ping/pong audio buffers + static zeros for silence */
static int16_t buf_ping[SAMPLES_PER_BUFFER];
static int16_t buf_pong[SAMPLES_PER_BUFFER];
static int16_t zeros[SAMPLES_PER_BUFFER];

/* Small fades to avoid ticks on transitions */
static inline void apply_fade(int16_t *buf, bool fade_in) {
    const int n = (int)(SAMPLE_RATE * 0.005f); /* ~5 ms worth of frames */
    const int m = (n < SAMPLES_PER_BLOCK) ? n : SAMPLES_PER_BLOCK;
    for (int i = 0; i < m; i++) {
        float g = fade_in ? (i + 1) / (float)m : (m - i) / (float)m;
        int32_t l = (int32_t)(buf[2 * i + 0] * g);
        int32_t r = (int32_t)(buf[2 * i + 1] * g);
        buf[2 * i + 0] = (int16_t)l;
        buf[2 * i + 1] = (int16_t)r;
    }
}

static inline void gen_sine_block(int16_t *buf) {
    for (uint32_t i = 0; i < SAMPLES_PER_BLOCK; i++) {
        uint32_t idx = (phase_acc >> 16) & (SINE_TABLE_SIZE - 1);
        int16_t s = sine_lut[idx];
        buf[2 * i + 0] = s; /* L */
        buf[2 * i + 1] = s; /* R */
        phase_acc += phase_step;
    }
}

/* ====== Feeder thread: only place that touches i2s_* ====== */
#define AUDIO_STACK_SIZE 2048
#define AUDIO_PRIO 0 /* higher than main */
K_THREAD_STACK_DEFINE(audio_stack, AUDIO_STACK_SIZE);
static struct k_thread audio_thread;

static atomic_t g_recoveries = ATOMIC_INIT(0);

/*
 * Workaround for NXP SAI: SAI_TxReset() (called inside i2s_configure /
 * recovery) clears MCR, which disables MCLK output.  Re-enable it after
 * every (re-)configure so the external DAC has a clock.
 */
static inline void sai_fixup_mclk(void)
{
#if DT_NODE_HAS_STATUS(DT_NODELABEL(sai0), okay)
    volatile uint32_t *mcr = (volatile uint32_t *)(0x50106000 + 0x100);
    *mcr |= (1u << 30); /* MCR.MOE = 1 */
#endif
}

static void audio_feeder(void *, void *, void *) {
    const struct device *i2s = DEVICE_DT_GET(I2S_DEV);
    if (!device_is_ready(i2s)) {
        printk("I2S not ready\n");
        return;
    }

    struct i2s_config cfg = {
        .word_size = SAMPLE_BIT_WIDTH,
        .channels = NUM_CHANNELS,
        .format = I2S_FMT_DATA_FORMAT_I2S,
        .options = I2S_OPT_BIT_CLK_MASTER | I2S_OPT_FRAME_CLK_MASTER,
        .frame_clk_freq = SAMPLE_RATE,
        .mem_slab = &tx_mem_slab,
        .block_size = BUFFER_SIZE_BYTES,
        .timeout = 50, /* ms: finite to avoid deadlocks */
    };

    if (i2s_configure(i2s, I2S_DIR_TX, &cfg) < 0) {
        printk("i2s_configure failed\n");
        return;
    }
    sai_fixup_mclk();

    /* Prefill 2 blocks so we hear something immediately */
    gen_sine_block(buf_ping);
    (void)i2s_buf_write(i2s, buf_ping, BUFFER_SIZE_BYTES);
    gen_sine_block(buf_pong);
    (void)i2s_buf_write(i2s, buf_pong, BUFFER_SIZE_BYTES);

    if (i2s_trigger(i2s, I2S_DIR_TX, I2S_TRIGGER_START) < 0) {
        printk("I2S START failed\n");
        return;
    }
    printk("Feeder running.\n");

    while (1) {
        void *payload = NULL;

        if (k_msgq_get(&audio_q, &payload, K_NO_WAIT) == 0) {
            /* App supplied a block */
        } else {
            payload = zeros; /* fallback to silence */
        }

        int r;
        do {
            r = i2s_buf_write(i2s, payload, BUFFER_SIZE_BYTES);
            if (r == -EAGAIN) k_sleep(K_MSEC(1));
        } while (r == -EAGAIN);

        if (r < 0) {
            atomic_inc(&g_recoveries);
            printk("audio: write=%d, recovering\n", r);
            (void)i2s_trigger(i2s, I2S_DIR_TX, I2S_TRIGGER_STOP);
            (void)i2s_trigger(i2s, I2S_DIR_TX, I2S_TRIGGER_DROP);
            if (i2s_configure(i2s, I2S_DIR_TX, &cfg) < 0) {
                printk("audio: reconfigure failed\n");
            }
            sai_fixup_mclk();
            (void)i2s_buf_write(i2s, zeros, BUFFER_SIZE_BYTES);
            (void)i2s_buf_write(i2s, zeros, BUFFER_SIZE_BYTES);
            (void)i2s_trigger(i2s, I2S_DIR_TX, I2S_TRIGGER_START);
        }

        k_yield(); /* keep it polite even though we outrank main */
    }
}

/* ====== main() = producer + CPU hog ======
 * Pattern: generate one block -> enqueue (or not) -> burn CPU until next
 * deadline. Toggles 1 s tone / 1 s silence; applies short fades on transitions.
 */
static inline void burn_until(uint64_t deadline_ms) {
    volatile uint32_t x = 1;
    while (k_uptime_get() < deadline_ms) {
        x = x * 1664525u + 1013904223u;
        if ((x & 0x7FFFu) == 0) k_yield();
    }
}

int main(void) {
    printk("Main produces; feeder fills gaps with silence (1s on/off)\n");

    memset(zeros, 0, sizeof(zeros));

    /* Ensure feeder outranks main */
    k_thread_priority_set(k_current_get(), 4);

    /* Start feeder */
    k_thread_create(&audio_thread, audio_stack,
                    K_THREAD_STACK_SIZEOF(audio_stack), audio_feeder, NULL,
                    NULL, NULL, AUDIO_PRIO, 0, K_NO_WAIT);

    const uint32_t block_ms =
        (SAMPLES_PER_BLOCK * 1000) / SAMPLE_RATE; /* ~21 ms */
    uint64_t next_deadline = k_uptime_get();

    bool tone_on = true;
    bool prev_tone_on =
        true; /* so first flip to silence can fade out if needed */
    bool pending_fade_out = false;
    uint64_t next_flip = k_uptime_get();

    int cur = 0;
    uint32_t last_log = 0;

    while (1) {
        uint64_t now = k_uptime_get();

        /* 1 s on/off toggle */
        if (now - next_flip >= 1000) {
            prev_tone_on = tone_on;
            tone_on = !tone_on;
            next_flip = now;
            if (!tone_on && prev_tone_on)
                pending_fade_out =
                    true; /* send one faded block before silence */
        }

        /* Produce exactly one block per period (or nothing during silence) */
        if (tone_on || pending_fade_out) {
            int16_t *b = (cur == 0) ? buf_ping : buf_pong;
            gen_sine_block(b);
            if (tone_on && !prev_tone_on)
                apply_fade(b, true); /* fade-in on first tone block */
            if (pending_fade_out)
                apply_fade(b, false); /* fade-out once, then silence */

            /* Try to enqueue; if full, drop (copy-based API, so no leak) */
            if (k_msgq_put(&audio_q, &b, K_NO_WAIT) != 0) {
                /* queue full → feeder will keep running with previous/zeros */
            }

            cur ^= 1;
            if (pending_fade_out) {
                pending_fade_out = false; /* next loop we truly go silent */
                prev_tone_on = false;
            } else {
                prev_tone_on = tone_on;
            }
        }

        /* Lightweight periodic status (IMMEDIATE logging; keep it sparse) */
        if ((uint32_t)(k_uptime_get_32() - last_log) > 1000) {
            last_log = k_uptime_get_32();
            printk("[main] tone=%d recoveries=%d\n", tone_on,
                   (int)atomic_get(&g_recoveries));
        }

        /* Burn CPU until the next block boundary */
        next_deadline += block_ms;
        if (next_deadline < now) next_deadline = now + block_ms;
        burn_until(next_deadline);
    }
}
