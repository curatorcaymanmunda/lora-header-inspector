# FLUX.3, MOSS-Image, and FIBO recognition

Added after the 0.6.2 release. These are header-only recognition rules, not
image-generation compatibility claims. No model or remote Python code is loaded.

## Primary implementation evidence inspected 2026-10-30

- [ComfyUI FLUX.3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/flux3/model.py):
  `Flux3` defines separate `content_mode_blocks.image` and `txt_mode_blocks`,
  with hidden size 3072. Both streams expose `q_proj` and `attn_out` linear layers.
- [OpenMOSS implementation](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/transformer/pipeline_moss_image.py):
  `MOSSImageTransformer` defines `shared_block_modulation` and
  `layers.N.proj.ffn_up_gate`, each with input width 3072.
- [ComfyUI FIBO 2.0 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/fibo2/model.py):
  `Fibo2` defines a Perceiver and main-layer cross-attention, both with
  `to_kv` context input width 7680. The independent anchors are
  `perceiver.layers.N.cross_attention.to_kv` and `layers.N.cross_attention.to_kv`.
- [ComfyUI model detection](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_detection.py)
  independently distinguishes the three architectures. Its full-checkpoint
  signatures cannot simply be copied to sparse adapters.
- [Bria FIBO](https://huggingface.co/briaai/FIBO) and
  [FIBO 2.0](https://huggingface.co/briaai/Fibo-2.0-dev) provide explicit model identities.

## Evidence policy and limits

| Label | Required structural evidence | Remaining limits |
|---|---|---|
| `Flux3` | Image and text mode-stream adapters, each with input width 3072 | FLUX.3 Image Klein BFL/Comfy spelling only; generic single-stream or Diffusers `mode_blocks` exports remain unresolved without metadata. No claim about the multimodal FLUX.3 backbone. |
| `MOSSImage` | Fused FFN gate plus shared block modulation adapters, each with input width 3072 | Exports targeting only attention or FFN omit a required anchor. Base and SFT are not distinguished structurally. |
| `FIBO2` | Perceiver and independent main-layer `cross_attention.to_kv` adapters, each with input width 7680 | ComfyUI spelling only; sparse attention-only adapters may omit one anchor. Dev/edit/turbo are not inferred from these shared modules. |
| `FIBO` | Explicit identity metadata only | Earlier FIBO generations have no new structural detector. Version metadata remains separate; no guessing from generic DiT keys. |

The detector handles dotted module names and flattened Kohya separators, and
uses the existing LoRA A/down input-width extraction. Synthetic fixtures cover
these recognized layouts; they are not sanitized examples of real trained LoRAs.
Missing anchors and wrong dimensions do not establish an exact identity.
Bare filename words `moss`, `fibo`, and `klein` are insufficient. Qualified
`MOSS-Image` and `Flux3` filename hints still require review under the existing
filename-only policy. Trusted identity metadata retains its existing behavior.
Conflicting structure and metadata remain blocking conflicts.

FLUX.3 Klein is not assigned the existing Flux.2-specific `Klein` variant, which
would incorrectly imply Flux.2. No inference about full model depth is made
from sparse block indices. No planner, authorization, journal, apply, undo,
or recovery code was changed for these additions.
