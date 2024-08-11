import os
import sys
import shutil
from pathlib import Path
from subprocess import Popen
from typing import Generator
from typing import Optional

import gradio as gr

current_path = Path(__file__).parent
current_path_str = str(current_path)
temp_path = current_path / "temp"
temp_path_str = str(temp_path)

current_value = os.environ.get("Path", '')
if current_path_str not in current_value:
    new_value = f"{current_path_str}{os.pathsep}{current_value}" if current_value else current_path_str
    os.environ["Path"] = new_value
os.environ["TEMP"] = temp_path_str

if temp_path.exists():
    shutil.rmtree(temp_path)
temp_path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, current_path_str)

from utils import Utils
from config import Config
from i18n import I18nAuto
from slicer import Slicer
from normalizer import Normalizer
from merger import Merger
from packer import Packer


class MainWebUI(object):

    def __init__(self) -> None:
        self.cfg = Config()
        self.utils = Utils()
        self.i18n = I18nAuto()
        self.normalizer = Normalizer()
        self.merger = Merger()
        self.packer = Packer()
        self.tran_webui_proc = None

        self.gr_main_title = "Homepage - G-SoMapper WebUI"
        self.gr_theme = self.cfg.gr_theme
        self.gr_max_size = self.cfg.gr_max_size
        self.gr_default_concurrency_limit = self.cfg.gr_default_concurrency_limit
        self.gr_is_inbrowser = self.cfg.gr_is_inbrowser
        self.gr_is_quiet = self.cfg.gr_is_quiet
        self.gr_is_share = self.cfg.gr_is_share
        self.gr_server_name = self.cfg.gr_server_name
        self.gr_main_webui_port = int(os.environ.get("main_webui_port", 23333))

        self.transcriber_webui_path = "transcriber_webui.py"
        self.transcriber_webui_cmd = ["python", self.transcriber_webui_path]

    def _open_slicer(
        self,
        input: Optional[tuple[str]],
        output: str,
        threshold: float,
        min_length: int,
        min_interval: int,
        hop_size: int,
        max_sil_kept: int
    ) -> Generator[tuple[str, dict[str, str | bool]], None, None]:
        slicer = Slicer(
            threshold,
            min_length,
            min_interval,
            hop_size,
            max_sil_kept
        )
        for res in slicer(input, output):
            yield res

    def _open_transcriber_webui(self, tran_webui_chk: bool) -> Generator[str, None, None]:
        if tran_webui_chk is True and self.tran_webui_proc is None:
            self.tran_webui_proc = Popen(self.transcriber_webui_cmd)
            open_msg = self.i18n(f"Transcriber WebUI 运行中：{self.transcriber_webui_path}")
            print(open_msg)
            yield open_msg
        elif tran_webui_chk is False and self.tran_webui_proc is not None:
            error_msg = self.i18n(self.utils.kill_proc(self.tran_webui_proc.pid))
            if error_msg != '':
                print(error_msg)
                yield error_msg

            self.tran_webui_proc = None

            close_msg = self.i18n(f"Transcriber WebUI 已关闭：{self.transcriber_webui_path}")
            print(close_msg)
            yield close_msg

    def __call__(self) -> None:
        with gr.Blocks(title=self.gr_main_title, theme=self.gr_theme) as app:
            gr.Markdown("# HomePage - G-SoMapper WebUI")
            gr.Markdown(self.i18n("##### [This repository is under MIT LICENSE protection](https://github.com/HaTiWinter/G-SoMapper) | Please follow the steps to start building your [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) training dataset:"))
            with gr.Tabs():
                with gr.TabItem(self.i18n("1. 准备音频")):
                    with gr.TabItem(self.i18n("1.1. 切分音频")):
                        gr.Markdown(self.i18n("##### 切分过长的视频或音频，输出 WAV 格式的 16 位 44100 Hz 单声道音频，防止后续 UVR 爆内存或爆显存。"))
                        with gr.Row():
                            with gr.Column():
                                slicer_input_path = gr.File(
                                    label=self.i18n("上传文件"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                slicer_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    slicer_threshold = gr.Slider(
                                        label=self.i18n("阈值（分贝）"),
                                        value=-24.0,
                                        minimum=-48.0,
                                        maximum=-0.1,
                                        step=0.1,
                                        interactive=True
                                    )
                                    with gr.Row():
                                        slicer_min_length = gr.Number(
                                            label=self.i18n("最短持续时间（毫秒）"),
                                            value=5000,
                                            step=1,
                                            precision=0,
                                            interactive=True
                                        )
                                        slicer_min_interval = gr.Number(
                                            label=self.i18n("最小切割间距（毫秒）"),
                                            value=100,
                                            step=1,
                                            precision=0,
                                            interactive=True
                                        )
                                    with gr.Row():
                                        slicer_hop_size = gr.Number(
                                            label=self.i18n("跳跃步长（毫秒）"),
                                            value=100,
                                            step=1,
                                            precision=0,
                                            interactive=True
                                        )
                                        slicer_max_sil_kept = gr.Number(
                                            label=self.i18n("最长静音时间（毫秒）"),
                                            value=100,
                                            step=1,
                                            precision=0,
                                            interactive=True
                                        )
                                with gr.Group():
                                    slicer_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                    open_slicer_btn = gr.Button(
                                        self.i18n("开始切分"),
                                        variant="primary",
                                        visible=True
                                    )
                                    open_slicer_btn.click(
                                        self._open_slicer,
                                        [
                                            slicer_input_path,
                                            slicer_output_path,
                                            slicer_threshold,
                                            slicer_min_length,
                                            slicer_min_interval,
                                            slicer_hop_size,
                                            slicer_max_sil_kept
                                        ],
                                        [slicer_info, open_slicer_btn],
                                    )
                    with gr.TabItem(self.i18n("1.2. 过滤音频")):
                        gr.Markdown(self.i18n("##### 过滤无关音频数据，优化音频质量。[点击此处下载最新版本的 UVR GUI](https://github.com/Anjok07/ultimatevocalremovergui/releases)"))
                        with gr.Group():
                            gr.Markdown(self.i18n("请打开 UVR GUI 过滤音频。以下 UVR 模型组合效果最好："))
                            gr.Markdown(self.i18n("**1. 提取人声：BS-Roformer-ViperX-1297 或 MDX23C-InstVoc HQ。**"))
                            gr.Markdown(self.i18n("**2. 去除混响：Reverb HQ。**"))
                            gr.Markdown(self.i18n("**3. 去除回声：UVR-De-Echo-Aggressive。**"))
                            gr.Markdown(self.i18n("**4. 去除噪音：UVR-DeNoise。**"))
                            gr.Markdown(self.i18n("按步骤调用模型即可获得最佳效果。注意："))
                            gr.Markdown(self.i18n("1. **BS-Roformer 系列模型**的速度和质量是 MDX23C 系列模型的 2 倍，显存仅占用 MDX23C 系列模型的一半， **但是 BS-Roformer 系列模型的使用需要安装 UVR GUI 测试版。暂时仅支持 Windows 平台。如果条件允许，请优先考虑使用 BS-Roformer 系列模型。** 推荐使用版本号更新的 BS-Roformer 模型，目前最新的是 BS-Roformer-ViperX-1297。"))
                            gr.Markdown(self.i18n("2. **UVR-De-Echo 系列模型**有三个档位，Normal、Aggressive，对应两种不同的回声强度，回声越大，选择的模型挡位也应当越高。Dereverb 挡位在 Aggressive 的基础上额外提供了去除混响功能，**但是 Reverb HQ 已经将混响去除，故不再重复此操作。**"))
                            gr.Markdown(self.i18n("3. **UVR-DeNoise 系列模型只推荐使用不带有 Lite 后缀的版本，即 UVR-DeNoise。**"))
                            gr.Markdown(self.i18n("4. **Reverb HQ 模型的效果比 UVR-De-Echo-Dereverb 好。**"))
                            gr.Markdown(self.i18n("5. **点击此处查看更详细的 UVR GUI 入门级教程：[图文版](https://www.bilibili.com/read/cv27499700) | [视频版](https://www.bilibili.com/video/BV1F4421c7qU)**"))
                    with gr.TabItem(self.i18n("1.3. 归一化音频")):
                        gr.Markdown(self.i18n("##### 均衡音频响度，优化音频质量；输出 WAV 格式的 16 位 32000 Hz 单声道音频，方便后续进一步处理。"))
                        with gr.Row():
                            with gr.Column():
                                norm_input_path = gr.File(
                                    label=self.i18n("上传音频"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                norm_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    gr.Markdown(self.i18n("执行 ITU-R BS.1770-4 标准"))
                                    with gr.Row():
                                        target_loud = gr.Slider(
                                            label=self.i18n("目标响度（分贝）"),
                                            minimum=-36.0,
                                            maximum=-6.0,
                                            value=-16.0,
                                            step=0.1,
                                            interactive=True
                                        )
                                        max_peak = gr.Slider(
                                            label=self.i18n("最大振幅（分贝）"),
                                            minimum=-12.0,
                                            maximum=-0.1,
                                            value=-1.0,
                                            step=0.1,
                                            interactive=True
                                        )
                                with gr.Group():
                                    norm_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                    open_norm_btn = gr.Button(
                                        self.i18n("开始归一化"),
                                        variant="primary",
                                        visible=True
                                    )
                                    open_norm_btn.click(
                                        self.normalizer,
                                        [
                                            norm_input_path,
                                            norm_output_path,
                                            target_loud,
                                            max_peak
                                        ],
                                        [norm_info, open_norm_btn]
                                    )
                with gr.TabItem(self.i18n("2. 准备标注")):
                    with gr.TabItem(self.i18n("2.1. 生成标注")):
                        gr.Markdown(self.i18n("##### 生成准确率较高的标注，需要后期手动校对。"))
                        with gr.Column():
                            tran_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                            tran_webui_chk = gr.Checkbox(
                                label=self.i18n("打开 Transcriber WebUI"),
                                value=False,
                                show_label=True,
                                interactive=True
                            )
                            tran_webui_chk.change(
                                self._open_transcriber_webui,
                                [tran_webui_chk],
                                [tran_info]
                            )
                    with gr.TabItem(self.i18n("2.2. 合并标注")):
                        gr.Markdown(self.i18n("##### 合并归一化后的音频和生成的标注，请注意，上传顺序要互相对应。"))
                        with gr.Row():
                            with gr.Column():
                                merger_audio_input_path = gr.File(
                                    label=self.i18n("上传音频"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                merger_subtitle_input_path = gr.File(
                                    label=self.i18n("上传标注"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                merger_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    merger_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                    open_merger_btn = gr.Button(
                                        self.i18n("开始合并"),
                                        variant="primary",
                                        visible=True
                                    )
                                    open_merger_btn.click(
                                        self.merger,
                                        [
                                            merger_audio_input_path,
                                            merger_subtitle_input_path,
                                            merger_output_path
                                        ],
                                        [merger_info, open_merger_btn]
                                    )
                    with gr.TabItem(self.i18n("2.3. 校对标注")):
                        gr.Markdown(self.i18n("##### 在 Aegisub 中手动校对 Transcriber 生成的标注，最终形成精确的标注。[点击此处下载最新版本的 Aegisub](https://aegisub.org/downloads)"))
                        with gr.Group():
                            gr.Markdown(self.i18n("请打开 Aegisub 校对标注。以下是一些注意事项："))
                            gr.Markdown(self.i18n("1. **请轴入实际文本，不要将文本标准化。** 不要轴入除当前语言的标点符号之外的任何符号，不需要将数字、时间、地点等文本标准化，请按照实际语言轴入实际文本。"))
                            gr.Markdown(self.i18n("2. **请确保每条标注的开始时间和结束时间准确无误。** 校对轴的质量好坏决定了标注的准确与否。"))
                            gr.Markdown(self.i18n("3. **请确保每条标注之间含有（半角）标点符号并且符合语义逻辑**，比如逗号、句号、感叹号等，标点符号使用不当会影响训练数据集的质量。"))
                            gr.Markdown(self.i18n("4. **请确保不同语言之间有合并节点分隔，每条标注之间缺少合并节点会导致后续数据无法合并。** 中文句号（。）、英文句号（.）、中文感叹号（！）、英文感叹号（!）、中文问号（？）、英文问号（?）将作为后续合并数据的合并节点。"))
                            gr.Markdown(self.i18n("5. **请尽量不要轴入无意义的音频，比如呼吸声、笑声、噪声等。** 过多的无意义的音频会影响训练数据集的质量。"))
                            gr.Markdown(self.i18n("6. **请尽量确保每个合并节点之间的持续时间在 3..10 秒之间。** 在参考音频的制作中，不在此范围内的段落将会被排除。"))
                            gr.Markdown(self.i18n("7. **点击此处查看更详细的 Aegisub 教程：[视频版](https://www.bilibili.com/video/BV1oK411T7kL/)**"))
                with gr.TabItem(self.i18n("3. 打包数据")):
                    gr.Markdown(self.i18n("##### 将归一化后的音频和校对后的标注打包成适用于 GPT-SoVITS 的训练数据集。请注意，上传顺序要相对应。"))
                    with gr.Row():
                            with gr.Column():
                                packer_audio_input_path = gr.File(
                                    label=self.i18n("上传音频"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                packer_subtitle_input_path = gr.File(
                                    label=self.i18n("上传标注"),
                                    type="filepath",
                                    file_count="multiple",
                                    interactive=True
                                )
                            with gr.Column():
                                packer_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    packer_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                    open_packer_btn = gr.Button(
                                        self.i18n("开始打包"),
                                        variant="primary",
                                        visible=True
                                    )
                                    open_packer_btn.click(
                                        self.packer,
                                        [
                                            packer_audio_input_path,
                                            packer_subtitle_input_path,
                                            packer_output_path
                                        ],
                                        [packer_info, open_packer_btn]
                                    )
                with gr.TabItem(self.i18n("4. 参考音频")):
                    with gr.TabItem(self.i18n("4.1. 情感识别")):
                        gr.Markdown(self.i18n("##### 施工中，请稍等……"))
            app.queue(max_size=self.gr_max_size, default_concurrency_limit=self.gr_default_concurrency_limit).launch(
                inbrowser=self.gr_is_inbrowser,
                quiet=self.gr_is_quiet,
                server_name=self.gr_server_name,
                server_port=self.gr_main_webui_port,
                share=self.gr_is_share
            )


if __name__ == "__main__":
    webui = MainWebUI()
    webui()
