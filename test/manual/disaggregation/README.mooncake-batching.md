# Cross-request Mooncake descriptor coalescing reproduction

Based on [SGLang PR #43995](https://github.com/sgl-project/sglang/pull/43995)
at `fd6cf87d3534dd4f97f181f91599ba3c3f6a5090`.

The current `_send_request_batch` concatenates each request's descriptors.
For two requests whose pages are adjacent in both the source and destination,
a single MHA layer with separate K and V buffers produces four descriptors:

```text
request 1: K[1] -> K[11], V[1] -> V[11]
request 2: K[2] -> K[12], V[2] -> V[12]
```

These can instead be represented by two ranges, one per buffer:

```text
K[1:3] -> K[11:13], V[1:3] -> V[11:13]
```

## Run

Use an environment with this checkout's SGLang runtime dependencies installed.
From the repository root:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python test/manual/disaggregation/repro_mooncake_descriptor_coalescing.py
```

The script calls the real `_send_request_batch`, `_plan_request_batch_chunk`,
layer mapping and `_transfer_data` methods. It constructs minimal in-memory
request/peer metadata, sets unified memory off, and substitutes CPU `memmove`
for the transport. No model, GPU, Mooncake engine or distributed service is
started. It does not import test fixtures or depend on local checkout names.
SGLang's normal import-time dependencies are still required.

Expected results at the pinned base:

| Case | Submissions | Submitted descriptors | Offline candidate descriptors |
|---|---:|---:|---:|
| Both ends contiguous | 1 | 4 | 2 |
| Source has a gap | 1 | 4 | 4 |
| Destination has a gap | 1 | 4 | 4 |

Both the original descriptors and the offline candidate are actually copied
and checked against the expected full destination buffers, including untouched
pages. The gap cases ensure that adjacency at only one end is insufficient.
The assertions intentionally describe the pinned implementation; update them
when adding coalescing to production.

## Suggested follow-up

Within compatible session/layout/buffer groups, coalesce ranges only when both
source and destination addresses are consecutive. Retain each request's
completion, cancellation and buffer-lifetime bookkeeping. The offline sort here
is only for these disjoint, synthetic ranges, not a production-safe algorithm
for arbitrary overlapping transfers or mixed layouts.

This is a deterministic descriptor-count/data-mapping reproduction, not a GPU
performance benchmark. It does not change production code, demonstrate a
correctness bug, or establish speedup. Real allocation traces and measurements
of planning cost, transport time and end-to-end latency are needed to determine
whether reducing descriptors pays for its own bookkeeping.

## Additional worker reproductions

The companion script reuses `TestMooncakeRequestBatching` from this checkout's
unit tests to exercise the real `transfer_worker`. This avoids duplicating the
worker's state/aux/notification fixture. It deliberately checks observations at
the pinned base, so it is a manual reproducer rather than a CI regression test
that would require these behaviors forever.

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python test/manual/disaggregation/repro_mooncake_request_batching.py
```

Use the runtime and unit-test dependencies for this checkout. Leave
`CUDA_VISIBLE_DEVICES` unset or set it to a nonempty value: the existing test
utilities do not support an empty value. All copied data is on the CPU.

- **Failed first sub-batch:** four requests, byte cap 16, two requests per
  sub-batch; injected first transport return -1. With batch size 1 there is one
  transport call, with size 8 there are two. The second call still sees an empty
  `failed_sessions` and writes its destination pages. All requests ultimately
  become Failed. This does not establish data corruption, premature release or
  false success. Stop further submissions to the failed session promptly while
  preserving deferred-release poisoning and failing unsubmitted members.
- **Singleton fallback:** eight requests, byte cap 8, only one request fits.
  Both modes submit eight times, but layer planning runs 8 versus 16 times.
  Reuse the prepared singleton plan or avoid futile preplanning. No timing or
  percentage speedup/regression is claimed by this script.
- **Completion publication:** four final chunks, byte cap 16. At the start of
  request 3's transfer, requests 1 and 2 are already Success with batch size 1,
  but neither is Success with size 8. Both modes finish correctly. Completing
  each bounded sub-batch's request bookkeeping before blocking on subsequent
  sub-batches could reduce this extra wait; ordering, cancellation, state/aux
  and buffer lifetimes need to remain correct. This is a latency tradeoff,
  not evidence of incorrect final results.

Actual returned values and descriptor counts are printed as JSON. Failures are
simulated and transport is CPU copying; this is not a network/GPU benchmark.
