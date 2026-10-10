<div align="center">

# 👁️ PhysicalEye-Decide

### Show it an image. Ask a question. Get a calibrated decision in one forward pass.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-PhysicalEye--Decide--35B-ffcc4d)](https://huggingface.co/masterset-ai/PhysicalEye-Decide-35B)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Space-ff9d00)](https://huggingface.co/spaces/masterset-ai/PhysicalEye-Decide-35B-demo)
[![Base model](https://img.shields.io/badge/Base-Darwin--35B--A3B-6f42c1)](https://huggingface.co/FINAL-Bench/Darwin-35B-A3B-Opus)
[![License](https://img.shields.io/badge/License-Apache%202.0-2ea44f)](LICENSE)

</div>

---

This repository holds the inference code and examples for **[PhysicalEye-Decide-35B](https://huggingface.co/masterset-ai/PhysicalEye-Decide-35B)**, a decision engine for images.

You give it an image, a question and the possible answers. It returns a probability for every answer. It does not generate text, so the answer is always one of your options.

| | |
|---|---|
| 🤗 **Model weights** | [masterset-ai/PhysicalEye-Decide-35B](https://huggingface.co/masterset-ai/PhysicalEye-Decide-35B) |
| 🖥️ **Live demo** | [masterset-ai/PhysicalEye-Decide-35B-demo](https://huggingface.co/spaces/masterset-ai/PhysicalEye-Decide-35B-demo) |
| 💻 **Code** | this repository |

## Results

| Test | Setting | Result |
|---|---|---|
| Egg inspection | Held-out test, 100 candling photos, good vs bad | **98.0%** |
| AI2D | 200 science diagram questions, 4 options | **87.0%** (chance 25%) |
| Speed | Median latency per decision, batched server, one B200 GPU | **0.051 s** |

## Quick start

```bash
git clone https://github.com/masterset-ai/PhysicalEye-Decide
cd PhysicalEye-Decide
pip install -r requirements.txt
python decide.py --image egg.jpg --instructions "Is this egg good or bad?" --options good bad
```

The first run downloads the weights from Hugging Face (about 69 GB). Use one GPU with 80 GB or more memory, or spread the model over several GPUs.

```python
from decide import Decider

d = Decider("masterset-ai/PhysicalEye-Decide-35B")
d.decide(
    images=["egg.jpg"],
    state="Candling photo: a light is shone through the egg in a dark room.",
    question={"type": "choice", "instructions": "Is this egg good or bad?",
              "criteria": {"good": "Normal, healthy egg", "bad": "Defective, spoiled or damaged egg"}},
)
# {"type": "choice", "choice": "good", "confidence": ..., "probabilities": {"good": ..., "bad": ...}}
```

## Question types

| Type | Fields | Returns |
|---|---|---|
| `choice` | `instructions`, `criteria`: option name to description (or `null`) | `choice`, `confidence`, `probabilities` |
| `noul` | `instructions`, optional `criteria`: `{"true": ..., "false": ...}` | `noul`: probability of yes |
| `score` | `instructions`, `criteria`: level descriptions, lowest first | `score`, `confidence`, `probabilities` |

Up to 4 images and up to 255 options per question.

## Examples

| File | What it shows |
|---|---|
| [`examples/choice_egg.py`](examples/choice_egg.py) | Pick one option |
| [`examples/yes_no_crack.py`](examples/yes_no_crack.py) | Yes or no |
| [`examples/score_damage.py`](examples/score_damage.py) | Score on a scale |
| [`examples/inspection_line.py`](examples/inspection_line.py) | Inspect a folder and send uncertain images to review |

## Base model and license

Built on [FINAL-Bench/Darwin-35B-A3B-Opus](https://huggingface.co/FINAL-Bench/Darwin-35B-A3B-Opus), a Qwen3.5 MoE based vision-language model. Code and weights are released under Apache-2.0; upstream license notices apply.
