import os
import socket
import sys
from pathlib import Path

import torch
import gradio as gr

gr_title = "G-SoMapper WebUI"
gr_theme = gr.themes.Default()
gr_max_size = int(os.environ.get("max_size", 1024))
gr_default_concurrency_limit = int(os.environ.get("default_concurrency_limit", 512))
gr_is_inbrowser = False if os.environ.get("is_inbrowser", "True").lower() == 'false' else True
gr_is_quiet = False if os.environ.get("is_quiet", "True").lower() == 'false' else True
gr_is_share = True if os.environ.get("is_share", "False").lower() == 'true' else False
gr_server_name = "0.0.0.0"
gr_webui_port = int(os.environ.get("webui_port", 23333))

ip_lookup_host = "1.1.1.1"
ip_lookup_port = 1
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ip_lookup_host, ip_lookup_port))
        return s.getsockname()[0]
    finally:
        s.close()

local_url = f"http://{get_local_ip()}:{gr_webui_port}"
os_name = sys.platform
python_ver = sys.version

if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(device=0)
    if (
        "16" in gpu_name
        and "V100" not in gpu_name.upper()
        or "P40" in gpu_name.upper()
        or "P10" in gpu_name.upper()
        or "1060" in gpu_name
        or "1070" in gpu_name
        or "1080" in gpu_name
    ):
        is_half = False
    else:
        is_half = True if os.environ.get("is_half", "False").lower() == 'true' else False
else:
    device = "cpu"
    is_half = False

uvr_path = Path("uvr") / "ultimatevocalremovergui" / "UVR.py"

funasr_large_model_path = "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
funasr_vad_model_path = "models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
funasr_punc_model_path = "models/punc_ct-transformer_cn-en-common-vocab471067-large"

class Config:
    def __init__(self):
        self.gr_title = gr_title
        self.gr_theme = gr_theme
        self.gr_max_size = gr_max_size
        self.gr_default_concurrency_limit = gr_default_concurrency_limit
        self.gr_is_inbrowser = gr_is_inbrowser
        self.gr_is_quiet = gr_is_quiet
        self.gr_is_share = gr_is_share
        self.gr_server_name = gr_server_name
        self.gr_webui_port = gr_webui_port

        self.local_url = local_url
        self.os_name = os_name
        self.python_ver = python_ver
        self.device = device

        self.is_half = is_half
        self.gpu_name = gpu_name

        self.uvr_path = uvr_path

        self.funasr_large_model_path = funasr_large_model_path
        self.funasr_vad_model_path = funasr_vad_model_path
        self.funasr_punc_model_path = funasr_punc_model_path
