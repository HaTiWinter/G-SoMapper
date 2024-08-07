import os
import sys
import psutil
import socket
import subprocess as subp
from subprocess import Popen

import torch
import gradio as gr

from i18n import I18nAuto

class Config:
    def __init__(self) -> None:
        self.i18n = I18nAuto()

        self.gr_main_title = "Homepage - G-SoMapper WebUI"
        self.gr_transcriber_title = "Transcriber - G-SoMapper WebUI"
        self.gr_theme = gr.themes.Default()
        self.gr_max_size = int(os.environ.get("max_size", 1024))
        self.gr_default_concurrency_limit = int(os.environ.get("default_concurrency_limit", 512))
        self.gr_is_inbrowser = False if os.environ.get("is_inbrowser", "True").lower() == 'false' else True
        self.gr_is_quiet = False if os.environ.get("is_quiet", "True").lower() == 'false' else True
        self.gr_is_share = True if os.environ.get("is_share", "False").lower() == 'true' else False
        self.gr_server_name = "0.0.0.0"
        self.gr_main_webui_port = int(os.environ.get("main_webui_port", 23333))
        self.gr_transcriber_webui_port = int(os.environ.get("transcriber_webui_port", 23334))

        self.kill_process_cmd_win = [
            "taskkill",
            "/F",
            "/T",
            "/PID"
        ]
        self.kill_process_cmd_linux_and_macos = ["kill", "-9"]
        self.transcriber_webui_path = "transcriber_webui.py"
        self.transcriber_webui_cmd = ["python", self.transcriber_webui_path]

        self.ip_lookup_host = "1.1.1.1"
        self.ip_lookup_port = 1
        self.local_ip = self._get_local_ip()
        self.main_local_url = f"http://{self.local_ip}:{self.gr_main_webui_port}"
        self.transcriber_local_url = f"http://{self.local_ip}:{self.gr_transcriber_webui_port}"
        self.os_name = sys.platform
        self.python_ver = sys.version

        self.device, self.gpu_count, self.gpu_names = self._get_device()
        self.is_half = self._get_is_half(self.gpu_names)

        self.total_cpu_cores = psutil.cpu_count(logical=False)
        self.model_path = "src/funasr/models/SenseVoiceSmall"
        self.vad_model_path = "src/funasr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.punc_model_path = "src/funasr/models/punc_ct-transformer_cn-en-common-vocab471067-large"

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self.ip_lookup_host, self.ip_lookup_port))
            return s.getsockname()[0]
        except:
            return "0.0.0.0"
        finally:
            s.close()

    def _get_device(self) -> tuple[str, int, list[str]]:
        if torch.cuda.is_available():
            device = "cuda"
            device_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        else:
            device = "cpu"
            device_count = 0
            gpu_names = []
        return device, device_count, gpu_names

    def _get_is_half(
        self,
        gpu_names: list[str]
    ) -> bool:
        gpu_names_upper = [name.upper() for name in gpu_names]
        if (
            "16" in gpu_names
            and "V100" not in gpu_names_upper
            or "P40" in gpu_names_upper
            or "P10" in gpu_names_upper
            or "1060" in gpu_names
            or "1070" in gpu_names
            or "1080" in gpu_names
        ):
            return False
        else:
            return True if os.environ.get("is_half", "False").lower() == 'true' else False

    def kill_process(
        self,
        pid: int
    ) -> str:
        if self.os_name == "win32":
            with Popen(
                self.kill_process_cmd_win + [str(pid)],
                stdout=subp.PIPE,
                stderr=subp.PIPE
            ) as proc:
                _, proc_err = proc.communicate()
                if proc.returncode != 0:
                    error_msg = self.i18n(f"无法终止进程：{pid}。")
                    print(error_msg)
                    print(proc_err.decode("UTF-8"))
                    return error_msg

                return ''
        elif self.os_name == "linux" or self.os_name == "darwin":
            with Popen(
                self.kill_process_cmd_linux_and_macos + [str(pid)],
                stdout=subp.PIPE,
                stderr=subp.PIPE
            ) as proc:
                _, proc_err = proc.communicate()
                if proc.returncode != 0:
                    error_msg = self.i18n(f"无法终止进程：{pid}。")
                    print(error_msg)
                    print(proc_err.decode("UTF-8"))
                    return error_msg

                return ''
        else:
            error_msg = self.i18n("终止进程错误：不支持的操作系统：{self.os_name}。")
            print(error_msg)
            return error_msg
