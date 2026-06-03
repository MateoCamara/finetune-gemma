# finetune-gemma

Fine-tune Google's **Gemma 2B** on a Spanish dataset with **QLoRA** (4-bit quantization plus
LoRA adapters), using Hugging Face `transformers`, `peft` and TRL's `SFTTrainer`. The script
runs a short training on
[`jhonparra18/spanish_billion_words_clean`](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean)
and pushes the resulting model to the Hugging Face Hub.

## Requirements

- A CUDA-capable **GPU** (training uses 4-bit `bitsandbytes` quantization).
- **PyTorch** built for your CUDA version — install it separately following the
  [official instructions](https://pytorch.org/get-started/locally/).
- A **Hugging Face account** with access to the gated **Gemma** model (accept the license at
  <https://huggingface.co/google/gemma-2b>) and an access token.
- Python 3.10+ and the packages in [`requirements.txt`](requirements.txt).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# add your Hugging Face token
cp .env.example .env               # then edit .env and set HUGGINGFACE_API_KEY
```

## Usage

```bash
python main.py
```

The script is organized into `# %%` cells, so you can also run it cell by cell in an IDE
(PyCharm / VS Code / Jupyter). It writes checkpoints to `gemma-2b-spanishbillionwords/` and,
when training finishes, calls `push_to_hub` to upload the model to your Hub account.

## Notes

- **Hardware:** `main.py` sets `CUDA_VISIBLE_DEVICES = "0,1"` with `device_map="auto"`. Adjust
  the device list to match your machine (e.g. `"0"` for a single GPU).
- **Hub target:** change the `push_to_hub(...)` repository id to your own namespace.
- **Hyperparameters:** the run is intentionally short (`max_steps=60`) as a demo; tune
  `max_steps`, batch size and the LoRA settings for a real fine-tune.

## License

The code in this repository is released under the [MIT License](LICENSE) © Mateo Cámara.
**Gemma** itself is distributed by Google under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); review and comply with them.
