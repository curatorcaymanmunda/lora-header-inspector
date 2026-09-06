# Structural signature rules

## Rule design

A structural rule should answer two questions separately:

1. What broad tensor layout is physically visible?
2. Is there enough distinctive evidence to name an exact base architecture?

A sparse LoRA may target one attention projection in one block. Absence of untargeted modules is not negative evidence, and the highest observed block number is not total model depth.

## Stable rule IDs

Examples include:

```text
sd.unet.cross-attention.2048
flux2.fused-single-qkv-mlp
chroma.distilled-guidance-layer
hunyuan.token-refiner
longcat.context-width-3584
zimage.width-signature
ideogram4.llm-indicator-modules
krea2.gqa-text-fusion-swiglu
anima.llm-adapter
minimax-h3.omni-dit
wan.block-layout
qwen-image.img-txt-mod
```

Rule IDs are intended for tests, fixture expectations, diagnostics, and future migrations. Human-readable messages may improve without changing a rule ID.

## Shared families

Some layouts intentionally remain broad:

- `DualSingleStreamDiT` can describe Flux.1, Flux.2, Chroma, LongCat, or HunyuanVideo portions.
- `SingleStreamSwiGLUDiT` can resemble Z-Image or Ideogram 4.
- `JointAttentionDiT` can resemble SD3 or Qwen-Image.
- `DualAttentionTransformerDiT` can cover several video-transformer implementations.
- `CosmosPredict2DiT` does not by itself prove Anima.

Metadata can settle an exact architecture when it agrees with the visible family.

## Negative controls

Every exact structural rule should include tests showing that its closest lookalike does not match. Examples:

- Flux.2 fused projection versus a generic dual/single-stream LoRA;
- Ideogram 4 width and LLM indicator versus Z-Image width and refiners;
- Krea 2 gated GQA/text fusion versus generic single-stream blocks;
- Anima LLM adapter versus a plain Cosmos-style denoiser;
- MiniMax H3 audio/video paths versus a generic fused-QKV transformer;
- SD3 recognized widths versus a 3072-wide shared joint-attention layout.

## Metadata and filenames

Metadata fields are independent evidence records. Identity fields receive stronger confidence than free-text titles. Filenames are word-bounded, lower priority, and cannot make an otherwise unsupported file eligible for automatic movement.

Common or overloaded tokens are deliberately inert without context:

```text
wan
anima
chroma
klein
schnell
qwen
bernie
minimax
```

## Adding a model

A durable addition needs:

1. primary implementation or model-definition evidence;
2. a list of distinctive and shared module names;
3. tensor dimensions where they are truly diagnostic;
4. exporter spelling variants;
5. positive tests;
6. negative lookalike tests;
7. a fixture;
8. documentation of cases that remain ambiguous.
