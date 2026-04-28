# GPT-SoVITS CPU 推理版

这是一个面向 Windows / Linux / macOS CPU 推理的 GPT-SoVITS 裁剪分支，只保留推理相关能力，目标是降低安装复杂度，并为后续 CPU 推理优化做准备。

[English](../../README.md)

## 这个仓库现在是什么

- 仅保留推理能力的 GPT-SoVITS 分支
- 设计目标是 CPU 推理，不再围绕 GPU 训练流程组织仓库
- 当前实际重点是 S2 `v2Pro` / `v2ProPlus` 路线，同时保留 `v1`、`v2`、`v2Pro`、`v2ProPlus` 的按版本推理权重下载

## 已删除内容

这个仓库已经移除了大部分上游训练与数据制作流程：

- 训练入口和训练专用工具
- 数据集切片、降噪、ASR、打标流程
- UVR5 和其他非推理 WebUI 工具

现在仓库的目标很单纯：更专注地跑 GPT-SoVITS CPU 推理。

## 当前保留的入口

- `webui.py`：最小推理启动器
- `GPT_SoVITS/inference_webui.py`：普通推理 WebUI
- `GPT_SoVITS/inference_webui_fast.py`：批量 / 快速推理 WebUI
- `api.py` 与 `api_v2.py`：推理 API

## 快速开始

### 1. 创建环境

```bash
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
```

### 2. 安装依赖并下载推理权重

CPU + ModelScope + `v2ProPlus` 示例：

```bash
bash install.sh --source ModelScope --version v2ProPlus
```

Windows PowerShell：

```powershell
.\install.ps1 -Source ModelScope -Version v2ProPlus
```

可选版本：

- `v1`
- `v2`
- `v2Pro`
- `v2ProPlus`
- `all`

### 3. 启动

推荐方式：

```bash
python webui.py
```

直接启动普通推理 WebUI：

```bash
python GPT_SoVITS/inference_webui.py
```

直接启动快速推理 WebUI：

```bash
python GPT_SoVITS/inference_webui_fast.py
```

## 说明

- 这个分支的目标是推理，不再支持训练工作流。
- 中文推理前处理仍然比英 / 日 / 韩更重，因为需要额外做 `g2pw` 和 BERT 文本特征。
- `install.sh` 和 `install.ps1` 现在只安装 CPU 版依赖，并按版本下载推理资源，不再整包下载全部预训练文件。
- `NLTK` 与 `OpenJTalk` 字典默认仍会下载。

## INT8 量化（降低内存）

本分支提供 G2PW、BERT-large、cnhubert 三个最大共享模型的 W8A8 INT8 量化版本。安装时加 `--int8` 可减少约 3 GB 内存。

### 安装

```bash
# INT8 模式（推荐内存受限环境使用）
bash install.sh --source ModelScope --version v2ProPlus --int8
```

### 各语种内存占用

INT8 模型采用懒加载——只有对应语种首次推理时才会加载到内存。

| 场景 | 启动后 | 首次推理峰值 | 说明 |
|---|---|---|---|
| **启动（未推理）** | ~2.2 GB | — | T2S + SoVITS + cnhubert + 框架 |
| **英文 / 日文 / 韩文** | ~2.4 GB | ~2.5 GB | 不需要 BERT 和 G2PW |
| **中文** | ~2.7 GB | ~3.5 GB | 首次中文请求时懒加载 BERT + G2PW |
| **多语种（auto）** | ~2.9 GB | ~3.7 GB | 首次使用时加载 LangSegmenter（fasttext 125MB） |

### 量化了什么

| 模型 | 原始大小 | INT8 | 精度 | 用途 |
|---|---|---|---|---|
| G2PW（BERT-base + 分类头） | 608 MB | 153 MB | 99.86% 准确率 | 中文多音字消歧 |
| BERT-large（22 层） | 1242 MB | 290 MB | cosine 0.983 | 中文文本特征 |
| cnhubert | 360 MB | 91 MB | cosine 0.985 | 音频特征提取 |
| **合计** | **2210 MB** | **534 MB** | | |

### 没有量化的

T2S 和 SoVITS **不做量化**，原因：
- 每个用户微调的语音模型权重不同
- T2S 自回归推理对量化误差累积敏感
- 保持 FP32（合计 ~670 MB）

### FP32 与 INT8 对比

| | FP32 | INT8 | 节省 |
|---|---|---|---|
| 启动内存 | ~5.2 GB | ~2.2 GB | **~3 GB (57%)** |
| 模型文件 | ~2.2 GB | ~534 MB | **~1.7 GB (75%)** |
| `transformers` 依赖 | 启动时必须加载 | 启动时不需要 | **~238 MB** |

### 技术细节

- 三个模型均使用手写 PyTorch forward（运行时不依赖 HuggingFace transformers）
- BERT-large 从 24 层裁剪到 22 层（TTS 不使用最后 2 层，无损）
- 量化使用 PyTorch Eager mode W8A8，敏感层保持 FP32/W8A32
- 校准数据：5285 条中文维基百科句子（G2PW）、560 条多语种音频（cnhubert）
- INT8 模型运行时自动检测——只要 `*_int8.pth` 文件存在就自动使用

## 速度摘要

在不改变各语种错误率、韵律、音色和音质的前提下，这个 CPU 分支目前已经做到：

- 中文端到端 `zh_pure`：`wall_sec 15.136264 -> 8.309471`，约 `-45.1%`
- 端到端：`wall_sec 10.431826 -> 7.046743`，约 `-32.4%`
- 预处理：`frontend_sec 0.867065 -> 0.569571`，约 `-34.3%`
- T2S：`t2s_sec 4.023657 -> 2.268571`，约 `-43.6%`
- VITS：`vits_sec 5.137876 -> 3.969286`，约 `-22.7%`

当前单阶段里，已经落地的最大收益是：

- 预处理：
  - 中文 warm 多句 BERT：`0.855s -> 0.131s`，约 `-84.7%`
  - 韩文 cold 前端：`0.791s -> 0.155s`，约 `-80.4%`
- T2S：`stable_batch_remap` 7-case 对照 `1.986543 -> 1.684446`，约 `-15.2%`
- VITS：`remove_weight_norm` 的 `vits_only` 对照 `4.256119 -> 4.102004`，约 `-3.6%`

## 这些加速是怎么做到的

这些收益不是靠句级缓存、量化、或者牺牲质量换来的，主要来自把 CPU 推理路径里原本很重的 Python 调度、重复准备、重复拷贝和无效冷启动逐步拆掉。

中文端到端收益会比总均值更明显，核心原因也很直接：中文原本是最重的一条路径，既要走 `g2pw`，又要做 BERT 文本特征，所以前端减负带来的收益会直接传导到整条链路。

### 明确没有做的

- 没有把 ONNX / ORT 接进当前主推理路径。这个方向做过 `dec-only ORT`、`flow + dec ORT`、更激进的图级 ORT 探索，但在当前机器和依赖组合下，没有拿到“完全无质量退化、同时还更快、更省”的结果，所以最终回到纯 PyTorch 主线。
- 中文路径没有为了提速去换低质量前端。`g2pw` 还在，中文 BERT 也还在；没有为了缩短耗时，改成像 `g2pm` 这类在复杂文本、尤其古文场景里更容易出明显 G2P 问题的轻量替代，也没有直接抛弃 BERT 特征。
- 没有把“二次切分长句、凑更大 batch”接进主路径。这个方向在 benchmark 里做过，但结论并不稳定，`repeats=3` 复验后总体还会变慢，所以没有拿来刷 README 里的主路径收益。

### 预处理 / 前端

- 英文前端重点优化的是冷启动。去掉了 `g2p_en.G2p` 的重型首包依赖，把 `wordsegment.load()` 改成懒加载，把 `nltk.pos_tag` 收缩到只在同形异音词场景触发，并绕开了 `inflect/typeguard` 导入期的额外成本。
- 韩文前端重点处理的是无关依赖串联冷启动。`g2pk2` 以前会顺手把 `nltk/cmudict` 一起拉起来，现在改成惰性 stub，纯韩文请求不再为英文词典买单。
- 中文前端最大的一刀是把纯中文多句 BERT 从逐句串行改成批量路径。现在会在纯中文、多句、无英文字母时直接走 `_preprocess_batch_zh()`，一次性完成 tokenizer、BERT forward 和 `word2ph` 对齐。
- 中文 `g2pw` 冷启动也做了多轮减负，包括把 `requests` 改成仅下载时导入、把 tokenizer 收缩到直接读 `tokenizer.json`、避免 `transformers` 自动分发链整包拉起。
- 中文分词/POS 这条线额外加了本地静态资产加载，把 `jieba_fast.posseg` 初始化的重复成本提前摊平，但没有做任何依赖输入文本的句级缓存。
- 非中文语种的零 BERT 特征也补了长度缓存，避免重复创建相同 shape 的全零张量。

### T2S

- 第一层收益来自把热路径里的纯 Python 开销删掉，包括推理循环里的 `tqdm` 和热打印；这类改动不碰模型数值，但在 CPU 上确实能省时间。
- 第二层收益来自 shrink 路径。原来 batch 内样本结束后会对整块未来容量一起 `index_select`，把大量未来根本不会用到的 token buffer、KV cache 和 mask 一起搬运。现在改成只搬运有效前缀，直接把 `index_select` 热点压下去。
- 第三层收益来自 `stable_batch_remap`。它不是“永不 shrink”，而是在保持 exact 的前提下，把 batch 内仍然活跃的行稳定重排，减少无意义的 batch compaction 抖动。
- 第四层收益来自直接处理 `addmm` 热点。现在主路径里已经把 `MLP`、`qkv_proj`、`out_proj` 这些高频线性层改成 exact-safe 的 hybrid 路径：
  `rows == 1` 保持原始 `F.linear`，其余走更贴近 CPU 热点的 `torch.addmm`。
- 这条线之所以能落地，是因为同时修了权重加载后的重建逻辑：`t2s_transformer` 会在 checkpoint load 完成后重建，保证预转置权重绑定的是真实加载后的参数，而不是初始化时的旧权重。

### VITS

- VITS 这边没有去动大模型结构，主思路是先把“每个 batch 都在重复做的准备工作”拿掉。
- 第一刀是在 `TTS.run()` 里做 run 级 runtime cache，只缓存当前 reference 相关、且与输入文本无关的对象，例如 `refer_audio_spec`、`sv_emb`、`prompt_semantic_tokens`、`prompt_phones`，避免同一次推理里按 batch 重复准备。
- 第二刀是把 `decode()` 里的参考条件编码抽出来，增加 `build_decode_condition()`，让 `ge / ge_text` 在 batch 循环前只算一次，后续 decode 直接复用。
- 第三刀是对非 vocoder 的 `Generator.dec` 直接执行 `remove_weight_norm()`，去掉推理阶段的重参数化开销。这一刀已经在 `vits_only` 口径下拿到稳定正收益。
- 整个 VITS 优化过程里，worklog 里还记录了多条试过但没保留的路线，例如 traced `dec`、更激进的 decode 布局实验。README 这里只保留已经真正落地主路径的部分。

### 推理加载与内存

- `t2s_only` benchmark 以前会先完整加载整条 `TTS()` pipeline，再把不用的对象 trim 掉，这会把峰值内存抬得很高。后面改成了真正的轻量加载路径，只初始化 `t2s` 需要的最小对象。
- 主推理路径里也对 `t2s` 的加载顺序做了重构，避免 checkpoint 还在内存里时又额外构建一整套转置权重副本。
- 更关键的一刀是推理路径完全不再构建 `self.h`。现在会直接从 state dict 重建 `t2s_transformer`，主入口只保留推理真正需要的那部分模块。
- 这条线已经接入真实推理路径，不只是 benchmark 技巧；它的作用主要是降低常驻 RSS 和峰值 RSS，避免 CPU 机器在跑推理时因为加载峰值过高而更容易卡顿或被系统回收。

## 上游项目与致谢

本项目基于并使用了以下上游项目代码：

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

下面保留上游及相关引用项目，避免丢失原始致谢信息。

## 引用项目

### 理论研究

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)

### 主模型 / 训练 / 声码器相关

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [SoVITS](https://github.com/voicepaw/so-vits-svc-fork)
- [GPT-SoVITS-beta](https://github.com/lj1995/GPT-SoVITS/tree/gsv-v2beta)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### 推理用文本前端

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### 继承自上游的工具引用

这些项目来自上游 GPT-SoVITS 的引用。虽然本裁剪分支已经删除了其中不少对应功能，但这里仍保留致谢。

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)
