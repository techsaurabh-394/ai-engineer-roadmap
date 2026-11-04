"""Toy inference-serving simulator: static vs continuous batching.

Standard library only, fully deterministic. Time is counted in decode steps.
In one step, every request that holds a GPU slot makes exactly one token.
(Prefill is ignored here to keep the lesson small.)

A request is a tuple (arrival_step, output_tokens).
Both schedulers return a list: finish_step for each request, in input order.
finish_step = the number of steps elapsed when its last token is done.
Latency of a request = finish_step - arrival_step.

Pass an empty list as `trace` to record (step, request_index) for every token.
"""

import math


def static_batching(requests, batch_size, trace=None):
    """Fixed batches. A batch runs until its LONGEST request is done, and all
    of its requests are released together. Requests are served first-come-first-served."""
    order = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    finish = [0] * len(requests)
    t = 0
    pos = 0
    while pos < len(order):
        first = order[pos]
        t = max(t, requests[first][0])  # wait for work if the GPU is idle
        batch = []
        while pos < len(order) and len(batch) < batch_size and requests[order[pos]][0] <= t:
            batch.append(order[pos])
            pos += 1
        duration = max(requests[i][1] for i in batch)
        if trace is not None:
            for step in range(duration):
                for i in batch:
                    if step < requests[i][1]:  # short requests sit idle after they finish
                        trace.append((t + step, i))
        t += duration
        for i in batch:
            finish[i] = t  # everyone waits for the longest
    return finish


def continuous_batching(requests, max_batch, trace=None):
    """Continuous batching: after every step, finished requests leave at once
    and waiting requests take the free slots."""
    order = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    remaining = {i: requests[i][1] for i in range(len(requests))}
    finish = [0] * len(requests)
    active = []
    pos = 0
    t = 0
    while pos < len(order) or active:
        while pos < len(order) and len(active) < max_batch and requests[order[pos]][0] <= t:
            active.append(order[pos])
            pos += 1
        for i in list(active):
            if trace is not None:
                trace.append((t, i))
            remaining[i] -= 1
            if remaining[i] == 0:
                finish[i] = t + 1
                active.remove(i)
        t += 1
    return finish


def percentile_nearest_rank(values, p):
    """Nearest-rank percentile: sort the values, take the item at
    rank = ceil(p/100 * n) (1-based). For n=6 and p=95, rank = ceil(5.7) = 6 -> the max."""
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def metrics(requests, finish, slots, trace):
    """Summary numbers for one run.

    throughput = total tokens / total steps (total steps = last finish step)
    utilization = tokens made / (slots * total steps), the share of GPU slot-steps doing work
    """
    latencies = [f - r[0] for f, r in zip(finish, requests)]
    total_steps = max(finish)
    tokens = len(trace)
    return {
        "mean_latency": sum(latencies) / len(latencies),
        "p95_latency": percentile_nearest_rank(latencies, 95),
        "throughput": tokens / total_steps,
        "utilization": tokens / (slots * total_steps),
        "total_steps": total_steps,
    }


def run(requests, slots):
    out = {}
    for name, fn in (("static", static_batching), ("continuous", continuous_batching)):
        trace = []
        finish = fn(requests, slots, trace)
        out[name] = {"finish": finish, **metrics(requests, finish, slots, trace)}
    return out


# Six requests, all arrive at step 0. Output lengths 2, 2, 2, 10, 2, 2.
SAMPLE = [(0, 2), (0, 2), (0, 2), (0, 10), (0, 2), (0, 2)]

if __name__ == "__main__":
    results = run(SAMPLE, slots=4)
    for name, r in results.items():
        print(
            f"{name:<10} finish={r['finish']} mean={r['mean_latency']:.2f} "
            f"p95={r['p95_latency']} throughput={r['throughput']:.2f} tok/step "
            f"util={r['utilization']:.1%}"
        )
