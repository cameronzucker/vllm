# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry of the sparse-MLA top-k index buffer for kpool indexers.

A kpool indexer selects ``index_topk // index_kpool`` pools and expands each
back to its ``index_kpool`` tokens, then appends the in-progress pool's tail
(``index_kpool - 1`` entries), so the row that reaches the attention kernel is
``index_topk + index_kpool - 1`` wide (2051 for GLM-5.3-Flash's 2048/4), padded
to the 128-column tiling (2176). Some sparse-MLA kernels are instantiated for a
fixed index width — the SM120 (compute capability 12.x) FlashInfer path reads
exactly 2048 entries per row — and cannot take the wider buffer. On those
devices the buffer is narrowed to the kernel width and the indexer selects as
many whole pools as still fit next to the tail: 511 pools + 3 tail entries in
2048 slots, the last slot staying masked (-1). The kernel's read count
(``index_topk``) and the checkpoint config are untouched.

Pure integer helpers, kept free of torch so they are unit-testable anywhere.
"""

# Index width the SM120 FlashInfer sparse-MLA decode kernels are built for.
SM120_SPARSE_INDEX_WIDTH = 2048

# Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
SPARSE_TOPK_BLOCK_N = 128


def natural_topk_buffer_width(index_topk: int, index_kpool: int) -> int:
    """Row width the kpool indexer wants: top-k tokens plus the pool tail,
    rounded up to the 128-column tiling."""
    tail = index_kpool - 1 if index_kpool > 1 else 0
    width = index_topk + tail
    return (
        (width + SPARSE_TOPK_BLOCK_N - 1) // SPARSE_TOPK_BLOCK_N * SPARSE_TOPK_BLOCK_N
    )


def topk_buffer_width(
    index_topk: int, index_kpool: int, kernel_width: int | None
) -> int:
    """Width of the ``topk_indices_buffer`` to allocate: the natural width,
    narrowed to ``kernel_width`` when the kernel cannot read a wider row."""
    natural = natural_topk_buffer_width(index_topk, index_kpool)
    if kernel_width is not None and natural > kernel_width:
        return kernel_width
    return natural


def kpool_effective_topk(topk_tokens: int, index_kpool: int, buffer_width: int) -> int:
    """Top-k the kpool indexer should select from, as whole pools, so that the
    expanded pools plus the tail fit in ``buffer_width`` slots. Equals
    ``topk_tokens`` whenever the buffer has room for the natural row."""
    if index_kpool <= 1:
        return min(topk_tokens, buffer_width)
    tail = index_kpool - 1
    max_pools = max((buffer_width - tail) // index_kpool, 0)
    return min(topk_tokens // index_kpool, max_pools) * index_kpool
