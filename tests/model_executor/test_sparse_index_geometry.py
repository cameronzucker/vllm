# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.model_executor.layers.sparse_index_geometry import (
    SM120_SPARSE_INDEX_WIDTH,
    kpool_effective_topk,
    natural_topk_buffer_width,
    topk_buffer_width,
)


def test_natural_width_pads_topk_plus_tail_to_128():
    assert natural_topk_buffer_width(2048, 4) == 2176  # 2051 -> 2176
    assert natural_topk_buffer_width(2048, 1) == 2048
    assert natural_topk_buffer_width(2044, 4) == 2048  # the hand-edited recipe


@pytest.mark.parametrize(
    ("index_topk", "kpool", "kernel_width", "expected"),
    [
        (2048, 4, SM120_SPARSE_INDEX_WIDTH, 2048),  # GLM-5.3-Flash on SM120
        (2048, 4, None, 2176),  # SM90: natural row, no narrowing
        (2048, 1, SM120_SPARSE_INDEX_WIDTH, 2048),  # DeepSeek-V3.2 style
        (2044, 4, SM120_SPARSE_INDEX_WIDTH, 2048),  # already fits
    ],
)
def test_buffer_width(index_topk, kpool, kernel_width, expected):
    assert topk_buffer_width(index_topk, kpool, kernel_width) == expected


@pytest.mark.parametrize(
    ("topk", "kpool", "buffer_width", "expected"),
    [
        (2048, 4, 2048, 2044),  # 511 pools + 3 tail = 2047 <= 2048
        (2048, 4, 2176, 2048),  # room for the natural row: unchanged
        (2044, 4, 2048, 2044),  # already pool-aligned and fitting
        (2048, 1, 2048, 2048),  # no pooling: whole buffer
        (2048, 16, 2048, 2032),  # 127 pools + 15 tail = 2047
    ],
)
def test_effective_topk_is_whole_pools_that_fit(topk, kpool, buffer_width, expected):
    eff = kpool_effective_topk(topk, kpool, buffer_width)
    assert eff == expected
    assert eff % kpool == 0
    tail = kpool - 1 if kpool > 1 else 0
    assert eff + tail <= buffer_width
