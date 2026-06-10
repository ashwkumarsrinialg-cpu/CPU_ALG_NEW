/*
 * scheduler.c
 * Predictive CPU Scheduler — Online MLR + SGD + Score-Based Aging
 * Ashwin Kumar Srinivasan, Amrita Vishwa Vidyapeetham, Coimbatore
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ─── Constants ──────────────────────────────────────────────────────────── */
#define NUM_PROCESSES   30
#define NUM_SEEDS       15
#define NUM_WORKLOADS    3
#define NUM_ALGORITHMS   5

#define LEARNING_RATE   0.01f
#define AGING_FACTOR    0.05f
#define RR_QUANTUM      4.0f        /* ms — Round Robin time quantum        */

/* Burst distribution parameters (uniform [lo, hi] ms) */
static const float BURST_LO[NUM_WORKLOADS] = {15.0f,  2.0f,  8.0f};
static const float BURST_HI[NUM_WORKLOADS] = {60.0f, 10.0f, 30.0f};
static const char *WORKLOAD_NAME[NUM_WORKLOADS] = {
    "CPU-bound", "I/O-bound", "Mixed"
};
static const char *ALGO_NAME[NUM_ALGORITHMS] = {
    "FCFS", "RoundRobin", "SJF", "SRTF", "MLR+SGD"
};

/* ─── Data structures ────────────────────────────────────────────────────── */
typedef struct {
    int   id;
    float arrival;          /* arrival time (ms)                            */
    float actual_burst;     /* ground-truth burst for this epoch            */
    float remaining;        /* used by SRTF                                 */
    float last_burst;       /* X1: most recent observed burst               */
    float avg_burst;        /* X2: running historical mean                  */
    float waiting_time;     /* cumulative wait in current epoch             */
    float predicted_burst;
    float final_score;
    int   burst_count;      /* how many bursts observed so far              */
    /* results */
    float finish_time;
    float turnaround;
    float awt_contribution; /* waiting time at finish                       */
} Process;

/* Per-seed per-algorithm metrics */
typedef struct {
    float awt;
    float atat;
    float ctx_switches;
} Metrics;

/* ─── Utilisation: uniform random float in [lo, hi] ─────────────────────────── */
static float rand_uniform(float lo, float hi) {
    return lo + ((float)rand() / (float)RAND_MAX) * (hi - lo);
}

/* ─── Make a fresh process queue ────────────────────────────────────── */
static void generate_queue(Process q[NUM_PROCESSES],
                            int workload_type, int seed)
{
    (void)seed; /* seed setting already done via srand() before calling */
    for (int i = 0; i < NUM_PROCESSES; i++) {
        q[i].id            = i + 1;
        q[i].arrival       = rand_uniform(0.0f, 60.0f);
        q[i].actual_burst  = rand_uniform(BURST_LO[workload_type],
                                          BURST_HI[workload_type]);
        q[i].remaining     = q[i].actual_burst;
        /* warm-start history: small jitter around the true burst          */
        q[i].last_burst    = q[i].actual_burst * rand_uniform(0.8f, 1.2f);
        q[i].avg_burst     = q[i].actual_burst * rand_uniform(0.8f, 1.2f);
        q[i].burst_count   = 1;
        q[i].waiting_time  = 0.0f;
        q[i].finish_time   = 0.0f;
        q[i].turnaround    = 0.0f;
        q[i].awt_contribution = 0.0f;
    }
}

/* Sort helpers */
static int cmp_arrival(const void *a, const void *b) {
    float da = ((Process*)a)->arrival;
    float db = ((Process*)b)->arrival;
    return (da > db) - (da < db);
}
static int cmp_burst(const void *a, const void *b) {
    float da = ((Process*)a)->actual_burst;
    float db = ((Process*)b)->actual_burst;
    return (da > db) - (da < db);
}
static int cmp_score(const void *a, const void *b) {
    float da = ((Process*)a)->final_score;
    float db = ((Process*)b)->final_score;
    return (da > db) - (da < db);
}

/* ─── Compute mean and std of an array ──────────────────────────────────── */
static void mean_std(const float arr[], int n, float *mean, float *std_dev) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += arr[i];
    *mean = s / (float)n;
    float sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = arr[i] - *mean;
        sq += d * d;
    }
    *std_dev = sqrtf(sq / (float)(n - 1));   /* sample std dev */
}

/* ─── Paired t-test (two-tailed) ────────────────────────────────────────── */
/*
 * Returns t-statistic. Degrees of freedom = n-1.
 * p-value is approximated using the Hill (1970) algorithm for the
 * t-distribution CDF — sufficient for reporting purposes.
 */
static float paired_t_stat(const float a[], const float b[], int n,
                            float *p_value)
{
    float diff[NUM_SEEDS];
    float mean_d = 0.0f;
    for (int i = 0; i < n; i++) {
        diff[i] = a[i] - b[i];
        mean_d += diff[i];
    }
    mean_d /= (float)n;

    float sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = diff[i] - mean_d;
        sq += d * d;
    }
    float sd = sqrtf(sq / (float)(n - 1));
    float t  = mean_d / (sd / sqrtf((float)n));

    /*
     * Two-tailed p-value via regularised incomplete beta function
     * approximation: p ≈ I(df/(df+t²), df/2, 1/2)
     * We use a simple numerical integration for small df (df=14 here).
     */
    int df = n - 1;
    float x = (float)df / ((float)df + t * t);
    /* Approximation: p ≈ 2 * BetaInc(x; df/2, 0.5) / Beta(df/2, 0.5)   */
    /* For df=14 we compute via series — good enough for reporting.        */
    /* Using the continued-fraction form (Abramowitz & Stegun 26.5.8):    */
    float a_beta = (float)df / 2.0f;
    float b_beta = 0.5f;
    /* Numerical integration of Beta(x; a, b) / Beta(a, b) via 1000 steps */
    int steps = 2000;
    float h    = x / (float)steps;
    float sum  = 0.0f;
    for (int k = 0; k <= steps; k++) {
        float xk = k * h;
        float w  = (k == 0 || k == steps) ? 0.5f : 1.0f; /* trapezoid   */
        if (xk > 0.0f && xk < 1.0f)
            sum += w * powf(xk, a_beta - 1.0f) * powf(1.0f - xk, b_beta - 1.0f);
    }
    sum *= h;
    /* Beta(a, b) via lgamma */
    float log_beta = lgammaf(a_beta) + lgammaf(b_beta) - lgammaf(a_beta + b_beta);
    float p_one_tail = sum / expf(log_beta);
    if (p_one_tail > 1.0f) p_one_tail = 1.0f;
    *p_value = 2.0f * p_one_tail;  /* two-tailed */
    return t;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 1: FCFS
 * ═══════════════════════════════════════════════════════════════════════════ */
static Metrics run_fcfs(Process q[NUM_PROCESSES]) {
    Process proc[NUM_PROCESSES];
    memcpy(proc, q, sizeof(proc));
    qsort(proc, NUM_PROCESSES, sizeof(Process), cmp_arrival);

    float time = 0.0f;
    float total_awt = 0.0f, total_atat = 0.0f;

    for (int i = 0; i < NUM_PROCESSES; i++) {
        if (time < proc[i].arrival) time = proc[i].arrival;
        float wait = time - proc[i].arrival;
        total_awt  += wait;
        time       += proc[i].actual_burst;
        total_atat += (time - proc[i].arrival);
    }
    Metrics m;
    m.awt         = total_awt  / NUM_PROCESSES;
    m.atat        = total_atat / NUM_PROCESSES;
    m.ctx_switches = (float)NUM_PROCESSES;
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 2: Round Robin
 * ═══════════════════════════════════════════════════════════════════════════ */
static Metrics run_rr(Process q[NUM_PROCESSES]) {
    Process proc[NUM_PROCESSES];
    memcpy(proc, q, sizeof(proc));
    qsort(proc, NUM_PROCESSES, sizeof(Process), cmp_arrival);

    float remaining[NUM_PROCESSES];
    float finish[NUM_PROCESSES];
    int   done[NUM_PROCESSES];
    for (int i = 0; i < NUM_PROCESSES; i++) {
        remaining[i] = proc[i].actual_burst;
        finish[i]    = 0.0f;
        done[i]      = 0;
    }

    float time = 0.0f;
    int   ctx  = 0;
    int   finished = 0;
    /* Simple RR: cycle through arrived processes */
    while (finished < NUM_PROCESSES) {
        int any = 0;
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (done[i] || proc[i].arrival > time) continue;
            if (remaining[i] <= 0.0f) continue;
            any = 1;
            ctx++;
            float run = (remaining[i] < RR_QUANTUM) ? remaining[i] : RR_QUANTUM;
            time        += run;
            remaining[i] -= run;
            if (remaining[i] <= 0.0f) {
                finish[i] = time;
                done[i]   = 1;
                finished++;
            }
        }
        if (!any) time += 0.1f;  /* advance past idle gap */
    }

    float total_awt = 0.0f, total_atat = 0.0f;
    for (int i = 0; i < NUM_PROCESSES; i++) {
        float tat = finish[i] - proc[i].arrival;
        total_atat += tat;
        total_awt  += tat - proc[i].actual_burst;
    }
    Metrics m;
    m.awt         = total_awt  / NUM_PROCESSES;
    m.atat        = total_atat / NUM_PROCESSES;
    m.ctx_switches = (float)ctx;
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 3: SJF (non-preemptive, with exponential smoothing prediction)
 * ═══════════════════════════════════════════════════════════════════════════ */
static Metrics run_sjf(Process q[NUM_PROCESSES]) {
    Process proc[NUM_PROCESSES];
    memcpy(proc, q, sizeof(proc));

    /* Exponential smoothing estimate: τ(n+1) = 0.5*t(n) + 0.5*τ(n)      */
    float est[NUM_PROCESSES];
    for (int i = 0; i < NUM_PROCESSES; i++)
        est[i] = proc[i].last_burst; /* τ(0) = last observed burst         */

    int   done[NUM_PROCESSES];
    memset(done, 0, sizeof(done));

    float time       = 0.0f;
    float total_awt  = 0.0f;
    float total_atat = 0.0f;
    int   ctx        = 0;

    for (int step = 0; step < NUM_PROCESSES; step++) {
        /* Find arrived, undone process with smallest estimate */
        int best = -1;
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (done[i] || proc[i].arrival > time) continue;
            if (best == -1 || est[i] < est[best]) best = i;
        }
        if (best == -1) {
            /* No arrived process — jump to next arrival */
            float next_arr = FLT_MAX;
            for (int i = 0; i < NUM_PROCESSES; i++)
                if (!done[i] && proc[i].arrival < next_arr)
                    next_arr = proc[i].arrival;
            time = next_arr;
            step--;
            continue;
        }
        ctx++;
        float wait = time - proc[best].arrival;
        total_awt += wait;
        time      += proc[best].actual_burst;
        total_atat += (time - proc[best].arrival);
        /* Update estimate using exponential smoothing */
        est[best] = 0.5f * proc[best].actual_burst + 0.5f * est[best];
        done[best] = 1;
    }
    Metrics m;
    m.awt         = total_awt  / NUM_PROCESSES;
    m.atat        = total_atat / NUM_PROCESSES;
    m.ctx_switches = (float)ctx;
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 4: SRTF (preemptive SJF)
 * ═══════════════════════════════════════════════════════════════════════════ */
static Metrics run_srtf(Process q[NUM_PROCESSES]) {
    Process proc[NUM_PROCESSES];
    memcpy(proc, q, sizeof(proc));

    float remaining[NUM_PROCESSES];
    float finish[NUM_PROCESSES];
    int   done[NUM_PROCESSES];
    float start_wait[NUM_PROCESSES]; /* time process first entered wait     */

    for (int i = 0; i < NUM_PROCESSES; i++) {
        remaining[i]   = proc[i].actual_burst;
        finish[i]      = 0.0f;
        done[i]        = 0;
        start_wait[i]  = proc[i].arrival;
    }

    float time     = 0.0f;
    int   finished = 0;
    int   ctx      = 0;
    int   prev     = -1;
    float step     = 0.1f;   /* simulation granularity (ms)                */

    /* Find earliest arrival to initialise time */
    float min_arr = FLT_MAX;
    for (int i = 0; i < NUM_PROCESSES; i++)
        if (proc[i].arrival < min_arr) min_arr = proc[i].arrival;
    time = min_arr;

    while (finished < NUM_PROCESSES) {
        /* Select arrived process with smallest remaining time */
        int best = -1;
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (done[i] || proc[i].arrival > time) continue;
            if (best == -1 || remaining[i] < remaining[best]) best = i;
        }
        if (best == -1) { time += step; continue; }

        if (best != prev) ctx++;
        prev = best;

        remaining[best] -= step;
        time            += step;

        if (remaining[best] <= 1e-4f) {
            finish[best]    = time;
            done[best]      = 1;
            finished++;
            prev = -1;
        }
    }

    float total_awt = 0.0f, total_atat = 0.0f;
    for (int i = 0; i < NUM_PROCESSES; i++) {
        float tat = finish[i] - proc[i].arrival;
        total_atat += tat;
        total_awt  += tat - proc[i].actual_burst;
    }
    Metrics m;
    m.awt         = total_awt  / NUM_PROCESSES;
    m.atat        = total_atat / NUM_PROCESSES;
    m.ctx_switches = (float)ctx;
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 5: MLR + SGD + Score-Based Aging  (the proposed algorithm)
 *
 *  Features:
 *    X1 = last_burst   (most recent observed burst)
 *    X2 = avg_burst    (running historical mean)
 *    X3 = queue_depth  (arrived processes not yet scheduled at this step)
 *
 *  Weights are re-initialised to priors at the start of each seed trial
 *  to simulate cold-start (pessimistic choice — see paper §5.1).
 * ═══════════════════════════════════════════════════════════════════════════ */
static Metrics run_mlr_sgd(Process q[NUM_PROCESSES]) {
    Process proc[NUM_PROCESSES];
    memcpy(proc, q, sizeof(proc));

    /* Weight priors (Section 3.2) */
    float b0 =  1.00f;
    float b1 =  0.50f;
    float b2 =  0.30f;
    float b3 = -0.05f;

    int   done[NUM_PROCESSES];
    memset(done, 0, sizeof(done));

    float time       = 0.0f;
    float total_awt  = 0.0f;
    float total_atat = 0.0f;
    int   ctx        = 0;

    for (int step = 0; step < NUM_PROCESSES; step++) {

        /* ── Phase A: count arrived, undone processes (X3) ────────────── */
        int queue_depth = 0;
        for (int i = 0; i < NUM_PROCESSES; i++)
            if (!done[i] && proc[i].arrival <= time) queue_depth++;

        /* If no process has arrived yet, jump to next arrival */
        if (queue_depth == 0) {
            float next_arr = FLT_MAX;
            for (int i = 0; i < NUM_PROCESSES; i++)
                if (!done[i] && proc[i].arrival < next_arr)
                    next_arr = proc[i].arrival;
            time = next_arr;
            step--;
            continue;
        }

        /* ── Phase A: predict and score all arrived, undone processes ─── */
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (done[i] || proc[i].arrival > time) continue;
            proc[i].predicted_burst = b0
                + b1 * proc[i].last_burst
                + b2 * proc[i].avg_burst
                + b3 * (float)queue_depth;
            proc[i].final_score = proc[i].predicted_burst
                - AGING_FACTOR * proc[i].waiting_time;
        }

        /* ── Phase B: select process with lowest score ─────────────────── */
        int best = -1;
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (done[i] || proc[i].arrival > time) continue;
            if (best == -1 || proc[i].final_score < proc[best].final_score)
                best = i;
        }

        ctx++;

        /* ── Phase C: execute ───────────────────────────────────────────── */
        float wait = time - proc[best].arrival;
        total_awt += wait;
        time      += proc[best].actual_burst;
        total_atat += (time - proc[best].arrival);

        /* ── Phase D: age remaining arrived processes ───────────────────── */
        for (int i = 0; i < NUM_PROCESSES; i++) {
            if (!done[i] && i != best && proc[i].arrival <= time)
                proc[i].waiting_time += proc[best].actual_burst;
        }

        /* ── Phase E: online SGD weight update ─────────────────────────── */
        float error = proc[best].actual_burst - proc[best].predicted_burst;
        b0 += LEARNING_RATE * error;
        b1 += LEARNING_RATE * error * proc[best].last_burst;
        b2 += LEARNING_RATE * error * proc[best].avg_burst;
        b3 += LEARNING_RATE * error * (float)queue_depth;

        /* ── Phase F: update telemetry for next cycle ───────────────────── */
        proc[best].avg_burst  = (proc[best].avg_burst + proc[best].actual_burst)
                                 / 2.0f;
        proc[best].last_burst = proc[best].actual_burst;
        done[best] = 1;
    }

    Metrics m;
    m.awt         = total_awt  / NUM_PROCESSES;
    m.atat        = total_atat / NUM_PROCESSES;
    m.ctx_switches = (float)ctx;
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    /* seeds 1..15 — fixed list for reproducibility */
    int seeds[NUM_SEEDS] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    /*
     * results[workload][algorithm][seed]
     * Store per-seed AWT for t-tests; also ATAT and ctx.
     */
    float awt_results [NUM_WORKLOADS][NUM_ALGORITHMS][NUM_SEEDS];
    float atat_results[NUM_WORKLOADS][NUM_ALGORITHMS][NUM_SEEDS];
    float ctx_results [NUM_WORKLOADS][NUM_ALGORITHMS][NUM_SEEDS];

    /* ── Run all experiments ─────────────────────────────────────────── */
    for (int w = 0; w < NUM_WORKLOADS; w++) {
        for (int s = 0; s < NUM_SEEDS; s++) {
            srand((unsigned int)seeds[s]);
            Process queue[NUM_PROCESSES];
            generate_queue(queue, w, seeds[s]);

            Metrics m[NUM_ALGORITHMS];
            m[0] = run_fcfs   (queue);
            m[1] = run_rr     (queue);
            m[2] = run_sjf    (queue);
            m[3] = run_srtf   (queue);
            m[4] = run_mlr_sgd(queue);

            for (int a = 0; a < NUM_ALGORITHMS; a++) {
                awt_results [w][a][s] = m[a].awt;
                atat_results[w][a][s] = m[a].atat;
                ctx_results [w][a][s] = m[a].ctx_switches;
            }
        }
    }

    /* ── Print results ───────────────────────────────────────────────── */
    printf("================================================================\n");
    printf("  Predictive CPU Scheduler — MLR+SGD  |  Ashwin Kumar Srinivasan\n");
    printf("  Amrita Vishwa Vidyapeetham, Coimbatore  |  June 2026\n");
    printf("================================================================\n");
    printf("Configuration: %d seeds x %d processes x %d workloads\n",
           NUM_SEEDS, NUM_PROCESSES, NUM_WORKLOADS);
    printf("Features: X1=last_burst  X2=hist_mean  X3=queue_depth\n");
    printf("RR quantum: %.1f ms   LR: %.3f   Aging gamma: %.3f\n\n",
           RR_QUANTUM, LEARNING_RATE, AGING_FACTOR);

    for (int w = 0; w < NUM_WORKLOADS; w++) {
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  WORKLOAD: %s  (burst U[%.0f, %.0f] ms)\n",
               WORKLOAD_NAME[w], BURST_LO[w], BURST_HI[w]);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("%-12s  %20s  %20s  %12s\n",
               "Algorithm", "AWT mean±std (ms)", "ATAT mean±std (ms)",
               "CtxSwitch");
        printf("%-12s  %20s  %20s  %12s\n",
               "------------", "--------------------", "--------------------",
               "------------");

        for (int a = 0; a < NUM_ALGORITHMS; a++) {
            float awt_m, awt_s, atat_m, atat_s, ctx_m, ctx_s;
            mean_std(awt_results [w][a], NUM_SEEDS, &awt_m,  &awt_s );
            mean_std(atat_results[w][a], NUM_SEEDS, &atat_m, &atat_s);
            mean_std(ctx_results [w][a], NUM_SEEDS, &ctx_m,  &ctx_s );
            printf("%-12s  %8.1f ± %6.1f  %8.1f ± %6.1f  %9.1f\n",
                   ALGO_NAME[a],
                   awt_m, awt_s,
                   atat_m, atat_s,
                   ctx_m);
        }

        /* Paired t-test: MLR+SGD (index 4) vs SJF (index 2) */
        float p_val;
        float t_stat = paired_t_stat(awt_results[w][2],   /* SJF         */
                                     awt_results[w][4],   /* MLR+SGD     */
                                     NUM_SEEDS, &p_val);

        float sjf_m,  sjf_s,  mlr_m,  mlr_s,  dummy;
        mean_std(awt_results[w][2], NUM_SEEDS, &sjf_m, &sjf_s);
        mean_std(awt_results[w][4], NUM_SEEDS, &mlr_m, &mlr_s);
        mean_std(ctx_results[w][2], NUM_SEEDS, &dummy, &dummy);

        float improvement = 100.0f * (sjf_m - mlr_m) / sjf_m;

        /* 95% CI for mean difference (SJF - MLR+SGD) */
        float diff[NUM_SEEDS];
        float mean_d = 0.0f;
        for (int s = 0; s < NUM_SEEDS; s++) {
            diff[s] = awt_results[w][2][s] - awt_results[w][4][s];
            mean_d += diff[s];
        }
        mean_d /= (float)NUM_SEEDS;
        float sq = 0.0f;
        for (int s = 0; s < NUM_SEEDS; s++) {
            float d = diff[s] - mean_d;
            sq += d * d;
        }
        float sd_d = sqrtf(sq / (float)(NUM_SEEDS - 1));
        /* t-critical for df=14, alpha=0.05, two-tailed ≈ 2.145 */
        float t_crit = 2.145f;
        float margin = t_crit * sd_d / sqrtf((float)NUM_SEEDS);
        float ci_lo  = mean_d - margin;
        float ci_hi  = mean_d + margin;

        printf("\n  Paired t-test (SJF vs MLR+SGD, AWT, df=14):\n");
        printf("  SJF AWT:     %.1f ms    MLR+SGD AWT: %.1f ms\n",
               sjf_m, mlr_m);
        printf("  Improvement: %.1f%%\n", improvement);
        printf("  t(%d) = %.2f,  p = %.4f  %s\n",
               NUM_SEEDS - 1, t_stat, p_val,
               (p_val < 0.05f) ? "** significant (p<0.05)" : "(not significant)");
        printf("  95%% CI for mean difference: [%.1f, %.1f] ms\n\n",
               ci_lo, ci_hi);
    }

    /* ── Aggregate across all workloads ─────────────────────────────── */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  AGGREGATE  (all %d workload types combined)\n",
           NUM_WORKLOADS);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("%-12s  %12s  %12s  %12s\n",
           "Algorithm", "Mean AWT(ms)", "Mean ATAT(ms)", "Mean CtxSw");
    printf("%-12s  %12s  %12s  %12s\n",
           "------------", "------------", "-------------", "----------");

    for (int a = 0; a < NUM_ALGORITHMS; a++) {
        float all_awt [NUM_WORKLOADS * NUM_SEEDS];
        float all_atat[NUM_WORKLOADS * NUM_SEEDS];
        float all_ctx [NUM_WORKLOADS * NUM_SEEDS];
        int idx = 0;
        for (int w = 0; w < NUM_WORKLOADS; w++)
            for (int s = 0; s < NUM_SEEDS; s++) {
                all_awt [idx] = awt_results [w][a][s];
                all_atat[idx] = atat_results[w][a][s];
                all_ctx [idx] = ctx_results [w][a][s];
                idx++;
            }
        float awt_m, awt_s, atat_m, atat_s, ctx_m, ctx_s;
        mean_std(all_awt,  idx, &awt_m,  &awt_s );
        mean_std(all_atat, idx, &atat_m, &atat_s);
        mean_std(all_ctx,  idx, &ctx_m,  &ctx_s );
        printf("%-12s  %12.1f  %13.1f  %10.1f\n",
               ALGO_NAME[a], awt_m, atat_m, ctx_m);
    }

    printf("\n================================================================\n");
    printf("GitHub: https://github.com/ashwkumarsrinialg-cpu\n");
    printf("================================================================\n");
    return 0;
}
