# Model recognition matrix

This table describes the intended evidence policy, not a claim that every sparse LoRA exposes every distinctive module.

| Label | Exact structural evidence currently used | Metadata assistance | Notes |
|---|---|---|---|
| SD 1.5 | SD UNet plus cross-attention context width 768 | Yes | Sparse conv-only LoRAs can remain unresolved. |
| SD 2.x | SD UNet plus cross-attention context width 1024 | Yes | Single text-encoder naming alone is insufficient. |
| SDXL | SD UNet plus context width 2048, or second text encoder with UNet layout | Yes | Pony, Illustrious, NoobAI, and Animagine are separate ecosystem fields. |
| SD3 / 3.5 | Native `joint_blocks` image/context spelling, or recognized Diffusers joint-attention widths | Yes | Shared 3072-wide joint-attention layouts remain ambiguous. |
| AuraFlow | Joint plus single transformer paths, or native double/single paths | Yes | Pony V7 is represented as Pony ecosystem on AuraFlow. |
| Flux.1 | Shared dual/single-stream family | Required for exact Flux.1 in many sparse exports | Generic Flux words are weak. |
| Flux.2 | Fused single-block QKV/MLP projection | Yes | Klein is a Flux.2 variant only with suitable context. |
| Chroma | `distilled_guidance_layer` | Yes | Bare filename word `chroma` is intentionally inert. |
| LongCat Image | 3584-wide context projection inside the dual/single family | Yes | Other Flux-like keys are shared. |
| HunyuanVideo | Token-refiner module path | Yes | Maximum observed block index is not treated as model depth. |
| Krea 2 | Text-fusion path or grouped-query gated attention plus SwiGLU widths | Yes | Supports Diffusers and native ComfyUI spellings. |
| Z-Image | Refiner path, or 3840/10240 single-stream width combination | Yes | Base and Turbo are checkpoint variants on the same architecture. |
| Ideogram 4 | LLM-condition/image-indicator paths, or 4608/12288 widths | Yes | Supports separate and fused QKV spellings. |
| Qwen-Image | Paired image and text modulation paths | Yes | Bare `qwen` does not mean Qwen-Image. |
| Anima | Anima LLM-adapter path | Yes | The shared Cosmos-style denoiser alone remains ambiguous. |
| ERNIE-Image | Self-attention plus its gated MLP spelling/widths | Yes | Not conflated with Bernini or a filename typo `Bernie`. |
| MiniMax H3 | Fused QKV, multimodal AdaLN, audio/video paths, and H3 widths | Yes | Bare MiniMax family naming remains unresolved. |
| Wan | Wan-style self-attention, cross-attention, and FFN layout | Yes | Structure alone may remain medium/ambiguous; version, task, and noise stage are qualifiers. |
| Bernini-R | Wan structure with Bernini-R model variant | Yes | Represented as a Wan-derived variant. |
| LTX | Shared dual-attention video-transformer family | Yes | Exact classification normally needs metadata because several video DiTs share key spelling. |

## Compatibility ecosystems

| Ecosystem | Allowed architecture scope |
|---|---|
| Pony | SD 1.5, SDXL, or AuraFlow depending on generation |
| Illustrious | SDXL |
| NoobAI | SDXL |
| Animagine | SDXL |

An ecosystem can imply a compatibility constraint, but it is not allowed to overrule contradictory structural evidence silently.

## Checkpoint variants and techniques

Checkpoint variants such as Z-Image Base, Z-Image Turbo, Flux Kontext, Flux Schnell, Flux Klein, Bernini-R, and Anima Aesthetic are recorded separately from techniques such as LCM, Lightning, Hyper, and DMD.

A generic `portrait_turbo.safetensors` may carry `Turbo` as a technique. An explicit `Z-Image-Turbo` identity carries `Turbo` as a checkpoint variant and does not duplicate the label in the technique field.
