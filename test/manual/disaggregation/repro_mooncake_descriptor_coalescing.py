"""Reproduce cross-request descriptor concatenation; CPU copies, no timing claims.

Run from the repository root with this checkout's python/ on PYTHONPATH.
Requires the cross-request batching implementation from SGLang PR #43995.
"""

import ctypes
import json
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager


def reproduce(src_pages, dst_pages):
    # One MHA layer, with separate K and V buffers. Each request owns one page.
    sources = [np.arange(32, dtype=np.int32) + i * 100 for i in range(2)]
    destinations = [np.full(32, -1, dtype=np.int32) for _ in sources]
    calls = []

    def copy_blocks(blocks):
        for src, dst, length in blocks:
            # Never dereference a descriptor unless both ends fit a matching pool.
            assert length > 0
            assert any(
                source.ctypes.data <= src
                and src + length <= source.ctypes.data + source.nbytes
                and target.ctypes.data <= dst
                and dst + length <= target.ctypes.data + target.nbytes
                for source, target in zip(sources, destinations, strict=True)
            )
            ctypes.memmove(dst, src, length)

    def transfer(session, src, dst, lengths):
        assert session == "peer"
        blocks = list(zip(src, dst, lengths, strict=True))
        copy_blocks(blocks)
        calls.append(blocks)
        return 0

    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
    manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
    manager.session_lock = Lock()
    manager.failed_sessions = set()
    manager.state_layout_rejections = {}
    manager.max_request_batch_bytes = 1024
    manager.max_request_batch_descriptors = 32
    manager.kv_args = SimpleNamespace(
        prefill_start_layer=0,
        kv_data_ptrs=[a.ctypes.data for a in sources],
        kv_item_lens=[a.itemsize for a in sources],
        kv_layer_ids=[],
    )
    manager.decode_kv_args_table = {
        "peer": SimpleNamespace(
            requires_dcp_relayout=False,
            dst_attn_tp_size=1,
            dst_kv_ptrs=[a.ctypes.data for a in destinations],
            dst_kv_layer_ids=[],
            dst_kv_item_len=4,
        )
    }
    manager.engine = SimpleNamespace(batch_transfer_sync=transfer)
    manager.request_status = {}
    manager.transfer_infos = {}
    chunks = []
    for room, (src, dst) in enumerate(zip(src_pages, dst_pages, strict=True), 1):
        manager.request_status[room] = KVPoll.WaitingForInput
        manager.transfer_infos[room] = {
            "peer": SimpleNamespace(
                is_dummy=False,
                mooncake_session_id="peer",
                dst_kv_indices=np.array([dst], dtype=np.int32),
                dst_device_kv_indices=None,
            )
        }
        chunks.append(
            TransferKVChunk(
                room=room,
                prefill_kv_indices=np.array([src], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=False,
                prefill_aux_index=None,
                state_indices=None,
            )
        )

    with patch(
        "sglang.srt.disaggregation.mooncake.conn.get_memory",
        return_value=SimpleNamespace(enable_unified_memory=False),
    ):
        results = manager._send_request_batch(chunks)
    assert results == {(id(chunk), "peer"): 0 for chunk in chunks}
    assert len(calls) == 1
    original = calls[0]

    expected = [np.full_like(a, -1) for a in destinations]
    for source, target, actual in zip(sources, expected, destinations, strict=True):
        target[dst_pages] = source[src_pages]
        np.testing.assert_array_equal(actual, target)

    # Offline candidate only: these requests have disjoint destination pages.
    # Group by the known K/V buffer pair; never merge across pool boundaries.
    candidate = []
    for source, target in zip(sources, destinations, strict=True):
        merged = []
        blocks = sorted(
            block
            for block in original
            if source.ctypes.data <= block[0] < source.ctypes.data + source.nbytes
        )
        for src, dst, length in blocks:
            if (
                merged
                and merged[-1][0] + merged[-1][2] == src
                and merged[-1][1] + merged[-1][2] == dst
            ):
                prev_src, prev_dst, prev_length = merged[-1]
                merged[-1] = (prev_src, prev_dst, prev_length + length)
            else:
                merged.append((src, dst, length))
        candidate.extend(merged)
        target.fill(-1)
    assert sum(b[2] for b in original) == sum(b[2] for b in candidate)
    copy_blocks(candidate)
    for actual, target in zip(destinations, expected, strict=True):
        np.testing.assert_array_equal(actual, target)
    return {
        "submissions": len(calls),
        "submitted_descriptors": len(original),
        "candidate_descriptors": len(candidate),
        "original_and_candidate_data_correct": True,
    }


if __name__ == "__main__":
    cases = {
        "both_contiguous": ([1, 2], [11, 12], 2),
        "source_gap": ([1, 3], [11, 12], 4),
        "destination_gap": ([1, 2], [11, 13], 4),
    }
    output = {}
    for name, (src, dst, count) in cases.items():
        result = reproduce(src, dst)
        assert result["submitted_descriptors"] == 4
        assert result["candidate_descriptors"] == count
        output[name] = result
    print(json.dumps(output, indent=2))
