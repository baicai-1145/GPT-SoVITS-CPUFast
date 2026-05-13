import os
import signal
import subprocess
import sys
from subprocess import DEVNULL, Popen

import gradio as gr

from config import (
    bert_path as default_bert_path,
    change_choices,
    cnhubert_path as default_cnhubert_path,
    get_weights_names,
    infer_device,
    is_half,
    is_share,
    python_exec,
    webui_port_infer_tts,
)
from tools.assets import css, js, top_html
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"] = language
i18n = I18nAuto(language=language)

_tts_process = None


def _kill_process_tree(proc: Popen | None):
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                f"taskkill /t /f /pid {proc.pid}",
                shell=True,
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except OSError:
        pass


def _status_text(opened: bool) -> str:
    if opened:
        return (
            i18n("推理 WebUI 已启动")
            + f" http://127.0.0.1:{webui_port_infer_tts}\n"
            + i18n("训练、数据集准备和标注相关功能已从此仓库移除。")
        )
    return i18n("当前仅保留推理功能。点击启动后会打开 GPT-SoVITS 推理 WebUI。")


def refresh_choices():
    sovits_update, gpt_update = change_choices()
    return sovits_update, gpt_update


def toggle_inference(bert_model_path, cnhubert_model_path, gpu_number, gpt_path, sovits_path):
    global _tts_process
    if _tts_process is not None:
        _kill_process_tree(_tts_process)
        _tts_process = None
        return (
            _status_text(False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    backend = "GPT_SoVITS/inference_webui_fast.py"
    env = os.environ.copy()
    if gpt_path:
        env["gpt_path"] = gpt_path
    else:
        env.pop("gpt_path", None)
    if sovits_path:
        env["sovits_path"] = sovits_path
    else:
        env.pop("sovits_path", None)
    env["cnhubert_base_path"] = cnhubert_model_path
    env["bert_path"] = bert_model_path
    env["_CUDA_VISIBLE_DEVICES"] = str(gpu_number)
    env["is_half"] = str(is_half)
    env["infer_ttswebui"] = str(webui_port_infer_tts)
    env["is_share"] = str(is_share)
    cmd = f'"{python_exec}" -s {backend} "{language}"'
    _tts_process = Popen(
        cmd,
        shell=True,
        env=env,
        start_new_session=(os.name != "nt"),
    )
    return (
        _status_text(True),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def on_close():
    global _tts_process
    _kill_process_tree(_tts_process)
    _tts_process = None


if not os.path.exists("GPT_SoVITS/text/G2PWModel"):
    cmd = f'"{python_exec}" -s GPT_SoVITS/download.py'
    proc = Popen(cmd, shell=True)
    proc.wait()

default_sovits, default_gpt = get_weights_names()
default_sovits_value = default_sovits[-1] if default_sovits else ""
default_gpt_value = default_gpt[-1] if default_gpt else ""
default_gpu = infer_device.index if getattr(infer_device, "type", "cpu") == "cuda" else 0

with gr.Blocks(title="GPT-SoVITS Inference Launcher", analytics_enabled=False, js=js, css=css) as app:
    gr.HTML(
        top_html.format(
            i18n("该裁剪版本仅保留推理相关功能。")
            + i18n("训练、数据集准备、打标、ASR、UVR5 和降噪入口已移除。")
        ),
        elem_classes="markdown",
    )
    with gr.Row():
        with gr.Column(scale=3):
            status = gr.Textbox(label=i18n("状态"), value=_status_text(False), lines=3)
            device_info = gr.Textbox(label=i18n("默认设备"), value=str(infer_device), interactive=False)
        with gr.Column(scale=1):
            open_btn = gr.Button(value=i18n("启动推理 WebUI"), variant="primary", visible=True)
            close_btn = gr.Button(value=i18n("关闭推理 WebUI"), variant="secondary", visible=False)
            refresh_btn = gr.Button(value=i18n("刷新权重列表"))
    with gr.Row():
        gpt_dropdown = gr.Dropdown(
            label=i18n("GPT 权重"),
            choices=default_gpt,
            value=default_gpt_value,
            allow_custom_value=True,
            interactive=True,
        )
        sovits_dropdown = gr.Dropdown(
            label=i18n("SoVITS 权重"),
            choices=default_sovits,
            value=default_sovits_value,
            allow_custom_value=True,
            interactive=True,
        )
    with gr.Row():
        bert_model_path = gr.Textbox(label=i18n("BERT 模型路径"), value=default_bert_path)
        cnhubert_model_path = gr.Textbox(label=i18n("CNHuBERT 模型路径"), value=default_cnhubert_path)
    with gr.Row():
        gpu_number = gr.Textbox(label=i18n("CUDA 设备号"), value=str(default_gpu))

    open_btn.click(
        toggle_inference,
        [bert_model_path, cnhubert_model_path, gpu_number, gpt_dropdown, sovits_dropdown],
        [status, open_btn, close_btn],
    )
    close_btn.click(
        toggle_inference,
        [bert_model_path, cnhubert_model_path, gpu_number, gpt_dropdown, sovits_dropdown],
        [status, open_btn, close_btn],
    )
    refresh_btn.click(refresh_choices, outputs=[sovits_dropdown, gpt_dropdown])

app.queue()
app.launch(server_name="0.0.0.0", server_port=9874, inbrowser=False, share=is_share)
on_close()
