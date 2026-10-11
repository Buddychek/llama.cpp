# Hybrid prefill workspace borrowing

Experimental and disabled by default. Borrowing reuses expert cache payload storage for temporary prefill activations and kernel scratch. It can reduce peak VRAM without changing the configured decode cache capacity or replacing arithmetic kernels.

## Usage

Add this option to an existing hybrid command before model loading:

```sh
--moe-hybrid on --moe-prefill-workspace borrow
```

To retain separate temporary allocations:

```sh
--moe-prefill-workspace separate
```

For a server `models.ini` preset:

```ini
moe-hybrid = on
moe-prefill-workspace = borrow
```

No environment variables are required. The option is independent of `--phase-aware-workspace`, expert profile selection, hardware split tuning and online residency adaptation. Ordinary execution is unchanged. See [hybrid execution](moe-hybrid.md) for the remaining model and cache options.

## Storage and phase transitions

The existing per-device grouped cache owner allocates stable payload backing. Bank strides, padding and slot counts come from the existing descriptors. Prepared prefill programs use the existing graph allocator and non-owning CUDA buffer views to reserve temporary storage within that backing.

Before scratch writes, the owner checks affected readers and adaptation copies, saves displaced expert identities and removes their residency claims. The saved identities are refill records; the canonical cache maps remain the residency authority. No expert placement may write into borrowed storage.

After outputs are published and scratch readers finish, the owner restores displaced payloads through the existing source transport. Ready residency is published only after copies complete. Route observations are retained, with placement maintenance deferred until the loan is returned. The cache is not cleared in full.

KV and recurrent state, caller inputs, persistent control storage and retained model leaves stay outside the loan. Temporary allocations that do not fit available backing use separate storage within the same executor. Other acquisition, return and refill failures are explicit; failed retirement retains backing rather than freeing live resources.

Borrowing does not add cache capacity or promise higher decode speed. Restoring displaced experts adds transfer work. The VRAM saving depends on the graph, microbatch, cache geometry and how much temporary storage fits the loan.

## Validation and limits

The existing prefill fixtures cover separate and fused gate/up layouts, Q5_K, IQ4_NL, Q4_K, Q8_0 and F32, repeated execution, graph reuse, exact restored payload bytes and route accounting. A four-window occurrence adaptation case checks learned placement and checked retirement.

Single-GPU serving checks include Flash Next with and without MTP, Nemotron, simultaneous requests and staggered prefill/decode requests. Completed checks recorded zero grouped fallback, rollback, prepare and finish errors, with clean teardown. A short fixed-profile prompt within one microbatch matched all 64 generated token IDs between separate and borrowed storage. Long generations and adaptive runs can diverge; these checks do not establish exact long-output parity.

A matched Flash Next observation with 8194 prompt tokens, 2048 generated tokens, microbatch 2048, cache 64 and MTP3 reduced sampled peak VRAM from 13739 to 12415 MiB. Decode measured 82.65 versus 82.58 tok/s and prefill was 2.69% lower. Outputs and MTP work differed between storage strategies, so this is a memory observation, not a statistical performance equivalence or speed claim.

Physical multi-GPU, Windows partial pinning and simultaneous independent execution contexts remain unqualified for this option. Conflicting owners fail closed rather than overwrite borrowed storage. Keep the default separate strategy where these additional qualifications are required.
