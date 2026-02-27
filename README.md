---
language:
- en
- zh
- ja
- ko
- fr
- de
- es
- pt
- it
- ru
license: apache-2.0
tags:
- text-generation
- writing
- content-generation
- zenlm
- zen
pipeline_tag: text-generation
---

# Zen Scribe 4B

**Professional content writing model fine-tuned for long-form generation, structured documents, and editorial quality output.**

Zen Scribe is a 4B parameter language model from [Zen LM](https://zenlm.org) optimized for writing tasks: blog posts, technical documentation, reports, creative writing, and structured content pipelines. It produces coherent, well-structured prose across extended contexts with consistent voice and style.

## Model Specs

| Property | Value |
|----------|-------|
| Parameters | 4B |
| Architecture | Transformer (decoder-only) |
| Context Window | 32,768 tokens |
| Output Format | Text |
| License | Apache 2.0 |
| HuggingFace | [zenlm/zen-scribe](https://huggingface.co/zenlm/zen-scribe) |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "zenlm/zen-scribe",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-scribe")

prompt = """Write a technical blog post introduction about vector databases:

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Use Cases

- **Technical documentation**: API references, guides, READMEs
- **Blog and editorial**: Long-form articles, opinion pieces, explainers
- **Business writing**: Reports, proposals, executive summaries
- **Creative writing**: Fiction, screenplays, narrative content
- **Structured output**: Templated content, form letters, product descriptions

## Content Pipeline Integration

Zen Scribe integrates with [Hanzo Flow](https://flow.hanzo.ai) for automated content pipelines:

```python
# Content pipeline: Brief → Draft → Edit → Publish
import hanzo

client = hanzo.Client()

draft = client.completions.create(
    model="zen-scribe",
    messages=[
        {"role": "system", "content": "You are a technical writer. Write clearly and concisely."},
        {"role": "user", "content": "Write a 500-word introduction to Kubernetes networking."}
    ],
    max_tokens=600,
)
print(draft.choices[0].message.content)
```

## MLX (Apple Silicon)

```bash
pip install mlx-lm
mlx_lm.generate --model zenlm/zen-scribe --prompt "Write an introduction to:" --max-tokens 500
```

## llama.cpp (CPU/GGUF)

```bash
llama-cli -m zen-scribe-q4_k_m.gguf -p "Write a technical blog post about:" -n 512
```

## Deployment

Zen Scribe is available via the [Hanzo API](https://api.hanzo.ai):

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zen-scribe",
    "messages": [{"role": "user", "content": "Write a product announcement for a new AI API:"}],
    "max_tokens": 400
  }'
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{zenlm2025zenscribe,
    title={Zen Scribe: Professional Content Writing Model},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-scribe}}
}
```

---

Part of the [Zen model family](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).

## Abliteration

Zen Scribe 4B weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream, as described in
[Maxime Labonne's methodology](https://huggingface.co/blog/mlabonne/abliteration).

**Method**: Directional ablation on the residual stream across all layers
**Source**: Zen MoDE (Mixture of Distilled Experts) architecture base
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.
