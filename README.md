# GPT-SoVITS CPU Inference Fork

Inference-only GPT-SoVITS fork focused on CPU deployment and CPU-side optimization on Windows, Linux, and macOS.

[中文简体](./docs/cn/README.md)

## What This Repo Is

- Inference-only fork of GPT-SoVITS.
- Designed around CPU usage rather than GPU-first features.
- Current practical focus is the S2 `v2Pro` / `v2ProPlus` path, while keeping versioned pretrained downloads for `v1`, `v2`, `v3`, `v4`, `v2Pro`, and `v2ProPlus`.

## What Was Removed

This repository no longer keeps most training and dataset-preparation features from upstream.

- Training entrypoints and training-only utilities
- Dataset slicing / denoise / ASR / labeling workflows
- UVR5 and other non-inference WebUI tools

The remaining goal is straightforward: run GPT-SoVITS inference on CPU with less installation friction and a smaller runtime surface.

## What Still Works

- `webui.py`: minimal inference launcher
- `GPT_SoVITS/inference_webui.py`: standard inference WebUI
- `GPT_SoVITS/inference_webui_fast.py`: batched / fast inference WebUI
- `api.py` and `api_v2.py`: inference APIs

## Quick Start

### 1. Create Environment

Use Miniconda or an existing Conda environment:

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
```

### 2. Install Dependencies and Download Inference Weights

CPU example with ModelScope and `v2ProPlus`:

```bash
bash install.sh --device CPU --source ModelScope --version v2ProPlus
```

Windows PowerShell:

```powershell
.\install.ps1 -Device CPU -Source ModelScope -Version v2ProPlus
```

Available versions:

- `v1`
- `v2`
- `v3`
- `v4`
- `v2Pro`
- `v2ProPlus`
- `all`

### 3. Launch

Recommended:

```bash
python webui.py
```

Direct inference WebUI:

```bash
python GPT_SoVITS/inference_webui.py
```

Fast inference WebUI:

```bash
python GPT_SoVITS/inference_webui_fast.py
```

## Notes

- This fork is aimed at CPU inference, not training.
- Chinese inference is still heavier than English / Japanese / Korean because text preprocessing needs extra frontend work such as `g2pw` and BERT features.
- `install.sh` and `install.ps1` now download inference assets by version instead of the full pretrained bundle.
- `NLTK` and `OpenJTalk` dictionary downloads remain enabled by default.

## Speed Summary

Without changing recognition behavior, prosody, speaker identity, or audio quality, this CPU fork currently delivers:

- Chinese end-to-end `zh_pure`: `wall_sec 15.136264 -> 8.309471`, about `-45.1%`
- End-to-end: `wall_sec 10.431826 -> 7.046743`, about `-32.4%`
- Preprocessing: `frontend_sec 0.867065 -> 0.569571`, about `-34.3%`
- T2S: `t2s_sec 4.023657 -> 2.268571`, about `-43.6%`
- VITS: `vits_sec 5.137876 -> 3.969286`, about `-22.7%`

Largest landed stage wins so far:

- Preprocessing:
  - Chinese warm multi-sentence BERT frontend: `0.855s -> 0.131s`, about `-84.7%`
  - Korean cold frontend path: `0.791s -> 0.155s`, about `-80.4%`
- T2S: `stable_batch_remap` 7-case comparison `1.986543 -> 1.684446`, about `-15.2%`
- VITS: `remove_weight_norm` `vits_only` comparison `4.256119 -> 4.102004`, about `-3.6%`

## How The Speedups Were Achieved

These gains do not come from sentence-level caching, quantization, or quality tradeoffs. The main work was removing CPU-side Python overhead, repeated preparation, repeated copies, and unnecessary cold-start costs from the real inference path.

Chinese shows an even larger end-to-end gain than the overall average because it started from the heaviest path: it pays for both `g2pw` and BERT text features, so frontend reductions propagate directly into total latency.

### What Was Deliberately Not Used

- ONNX / ORT was not kept in the main inference path. `dec-only ORT`, `flow + dec ORT`, and larger graph-level ORT experiments were all tried, but on the current machine and dependency stack they did not produce a solution that was simultaneously quality-safe, faster, and lighter than the PyTorch path, so the runtime stayed on pure PyTorch.
- The Chinese path was not simplified by dropping quality-critical frontend pieces. `g2pw` stayed, Chinese BERT stayed, and the project did not switch to lighter but lower-quality replacements such as `g2pm`, which are more likely to cause noticeable G2P errors on harder text, especially literary or classical material.
- The main path also does not rely on secondary splitting just to manufacture larger batches. That direction was explored in benchmark-only form, but the results did not stay stable, and the `repeats=3` verification did not justify moving it into the runtime path.

### Preprocessing / Frontend

- English frontend work focused on cold start. Heavy first-request dependencies around `g2p_en.G2p` were reduced, `wordsegment.load()` was made lazy, `nltk.pos_tag` was narrowed to heteronym cases, and import-time overhead from `inflect/typeguard` was trimmed.
- Korean frontend work removed an unrelated cold-start chain. `g2pk2` used to pull in `nltk/cmudict` even for pure Korean input; that path is now stubbed lazily so Korean requests no longer pay for the English dictionary on first use.
- The biggest Chinese frontend win came from changing pure-Chinese multi-sentence BERT extraction from sentence-by-sentence serial execution to a batched path. The current path batches tokenization, BERT forward, and `word2ph` alignment for pure Chinese multi-sentence requests.
- Chinese `g2pw` cold start was also reduced by lazily importing `requests`, shrinking tokenizer initialization down to direct `tokenizer.json` loading, and avoiding the larger `transformers` auto-dispatch chain during startup.
- Chinese segmentation / POS loading was further reduced with local static assets so `jieba_fast.posseg` initialization does less repeated work, without introducing any sentence-level cache tied to user input.
- Non-Chinese zero-BERT paths also gained simple length-based zero-tensor reuse so repeated all-zero feature allocation does less work.

### T2S

- The first layer of gains came from removing pure Python overhead from the hot path, including `tqdm` and hot-loop prints during decoding. These changes do not alter model numerics, but they do matter on CPU.
- The next layer came from the shrink path. Previously, when some rows in a batch finished early, the code copied whole future-capacity buffers with `index_select`, including token buffers, KV caches, and masks that were never going to be used. The current path compacts only the valid prefix.
- Another landed gain is `stable_batch_remap`. It is not a “never shrink” hack. It keeps exact behavior while stabilizing how active rows are remapped inside the batch, which reduces unnecessary compaction churn.
- The deeper T2S gains came from addressing the actual `addmm` hotspots. The main path now uses exact-safe hybrid linear execution for high-frequency layers such as `MLP`, `qkv_proj`, and `out_proj`: `rows == 1` keeps the original `F.linear`, while larger cases use a more CPU-friendly `torch.addmm` path.
- That backend work only became safe after fixing the load path as well. `t2s_transformer` is rebuilt after checkpoint loading so cached transposed weights are bound to the real loaded parameters instead of the pre-load initialization weights.

### VITS

- VITS optimization stayed conservative. The main strategy was to remove work that was being repeated for every batch instead of rewriting the model structure.
- The first landed step was a run-level runtime cache in `TTS.run()` for reference-side objects that do not depend on the current input text, such as `refer_audio_spec`, `sv_emb`, `prompt_semantic_tokens`, and `prompt_phones`.
- The second landed step moved decode-condition preparation out of repeated decode calls. `build_decode_condition()` lets `ge / ge_text` be computed once before the batch loop and then reused across decode calls.
- The third landed step applies `remove_weight_norm()` directly to non-vocoder `Generator.dec`, removing weight-norm reparameterization overhead during inference. This is the main clearly logged stage-local VITS win currently kept in the codebase.
- The worklog also records more aggressive VITS experiments that were tested but not kept, such as traced `dec` and more aggressive decode layout variants. The README only describes the parts that actually landed.

### Load Path And Memory

- `t2s_only` benchmark loading used to instantiate the full `TTS()` pipeline and then trim unused objects, which pushed peak memory far too high. That was replaced with a true lightweight load path that initializes only the minimum T2S-side objects.
- The main inference load path was also reordered to avoid rebuilding large transposed-weight structures while the checkpoint dictionary was still alive in memory.
- The biggest memory-side change is that inference no longer builds `self.h` at all on the main T2S path. The runtime now rebuilds `t2s_transformer` directly from the state dict and only keeps what inference actually uses.
- This is not benchmark-only machinery. It is wired into the real inference path and mainly helps reduce steady-state RSS and peak RSS so CPU machines are less likely to stall or be reclaimed under memory pressure.

## Upstream and Credits

This project is based on and uses code from:

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

This fork keeps upstream credits and referenced projects below.

## Referenced Projects

### Theoretical Research

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### Main Model / Training / Vocoder Related

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SoVITS](https://github.com/voicepaw/so-vits-svc-fork)
- [GPT-SoVITS-beta](https://github.com/lj1995/GPT-SoVITS/tree/gsv-v2beta)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### Text Frontend for Inference

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### Inherited Upstream Tool References

These projects were referenced by upstream GPT-SoVITS. Some related modules are removed in this inference-only fork, but credits are preserved here.

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
