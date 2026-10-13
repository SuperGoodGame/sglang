"""Reproduce PR #43995 worker behavior using its existing CPU-copy fixture.

These assertions describe the pinned PR, not the desired production behavior.
No GPU transport, timing or model inference is performed.
"""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "batch_fixture",
    REPO / "test/registered/unit/disaggregation/test_mooncake_transfer_batching.py",
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
assert (
    Path(fixture.MooncakeKVManager.transfer_worker.__code__.co_filename)
    .resolve()
    .is_relative_to(REPO)
)


def duplicate_planning(batch_size):
    t = fixture.TestMooncakeRequestBatching()
    t.setUp()
    try:
        m = t._manager(batch_size=batch_size)
        m.max_request_batch_bytes = 8  # one request fits; two cannot fit together
        plans = []
        original = m._get_kvcache_layer_params

        def counted(*a, **kw):
            plans.append(1)
            return original(*a, **kw)

        m._get_kvcache_layer_params = counted
        chunks = [t._chunk(m, i, [i], [i + 10]) for i in range(1, 9)]
        t._run(m, chunks)
        t._assert_pages("peer", list(range(1, 9)), list(range(11, 19)))
        assert len(t.calls) == 8
        assert len(plans) == (8 if batch_size == 1 else 16)
        return {
            "batch_size": batch_size,
            "submissions": len(t.calls),
            "planning_calls": len(plans),
        }
    finally:
        t.doCleanups()


def completion_barrier(batch_size):
    t = fixture.TestMooncakeRequestBatching()
    t.setUp()
    try:
        m = t._manager(batch_size=batch_size)
        m.max_request_batch_bytes = 16  # split four requests into two pairs
        chunks = [t._chunk(m, i, [i], [i + 10], last=True) for i in range(1, 5)]
        observed = []
        third_source = t.sources[0].ctypes.data + 3 * 4

        def before_transfer(manager):
            if t.calls[-1][1][0] == third_source:
                observed.append(
                    [
                        i
                        for i in (1, 2)
                        if manager.request_status[i] == fixture.KVPoll.Success
                    ]
                )

        t.before_transfer = before_transfer
        t._run(m, chunks)
        assert observed == ([[1, 2]] if batch_size == 1 else [[]])
        assert all(v == fixture.KVPoll.Success for v in m.request_status.values())
        return {
            "batch_size": batch_size,
            "first_pair_completed_before_third_request_transfer": observed[0],
        }
    finally:
        t.doCleanups()


def failure_submissions(batch_size):
    test = fixture.TestMooncakeRequestBatching()
    test.setUp()
    try:
        manager = test._manager(batch_size=batch_size)
        manager.max_request_batch_bytes = 16  # two one-page requests per sub-batch
        chunks = [test._chunk(manager, i, [i], [i + 10]) for i in range(1, 5)]
        test.failed_submission = 1
        snapshots = []
        test.before_transfer = lambda m: snapshots.append(
            {
                "failed_sessions": sorted(m.failed_sessions),
                "statuses": {k: int(v) for k, v in m.request_status.items()},
            }
        )
        test._run(manager, chunks)
        result = {
            "batch_size": batch_size,
            "transport_calls": len(test.calls),
            "descriptor_counts": [len(c[3]) for c in test.calls],
            "snapshots": snapshots,
            "statuses": {k: int(v) for k, v in manager.request_status.items()},
            "written_destination_indices": [
                i for i in range(11, 15) if test.destinations["peer"][0][i] != -1
            ],
            "poisoned_rooms": sorted(manager._deferred_ack_poisoned_rooms),
        }
        assert all(
            status == fixture.KVPoll.Failed
            for status in manager.request_status.values()
        )
        assert len(test.calls) == (1 if batch_size == 1 else 2)
        assert result["written_destination_indices"] == (
            [] if batch_size == 1 else [13, 14]
        )
        return result
    finally:
        test.doCleanups()


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "failure_submissions": [failure_submissions(n) for n in (1, 8)],
                "duplicate_planning": [duplicate_planning(n) for n in (1, 8)],
                "completion_barrier": [completion_barrier(n) for n in (1, 8)],
            },
            indent=2,
        )
    )
