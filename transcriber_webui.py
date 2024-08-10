import os
import sys
from pathlib import Path

import gradio as gr

current_path = Path(__file__).parent
current_path_str = str(current_path)

sys.path.insert(0, current_path_str)

from config import Config
from i18n import I18nAuto
from label.transcriber import AudioTranscriber


class TranscriberWebUI(object):

    def __init__(self) -> None:
        self.cfg = Config()
        self.i18n = I18nAuto()
        self.tran = AudioTranscriber(lang="auto")

        self.gr_transcriber_title = "Transcriber - G-SoMapper WebUI"
        self.gr_theme = self.cfg.gr_theme
        self.gr_max_size = self.cfg.gr_max_size
        self.gr_default_concurrency_limit = self.cfg.gr_default_concurrency_limit
        self.gr_is_inbrowser = self.cfg.gr_is_inbrowser
        self.gr_is_quiet = self.cfg.gr_is_quiet
        self.gr_is_share = self.cfg.gr_is_share
        self.gr_server_name = self.cfg.gr_server_name
        self.gr_transcriber_webui_port = int(os.environ.get("transcriber_webui_port", 23334))

    def __call__(self) -> None:
        with gr.Blocks(title=self.gr_transcriber_title, theme=self.gr_theme) as app:
            gr.Markdown(self.i18n("# Transcriber - G-SoMapper WebUI"))
            gr.Markdown(self.i18n("##### [此项目受 MIT LICENSE 保护](https://github.com/HaTiWinter/G-SoMapper) | 及时关闭 Transcriber WebUI 可以减少显存占用："))
            with gr.Row():
                with gr.Column():
                    tran_input_path = gr.File(
                        label=self.i18n("上传音频"),
                        type="filepath",
                        file_count="multiple",
                        interactive=True,
                    )
                with gr.Column():
                    tran_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                    with gr.Group():
                        tran_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                        open_tran_btn = gr.Button(self.i18n("开始生成"), variant="primary", visible=True)
                        open_tran_btn.click(
                            self.tran.Transcriber,
                            [tran_input_path, tran_output_path],
                            [tran_info, open_tran_btn]
                        )
            app.queue(max_size=self.gr_max_size, default_concurrency_limit=self.gr_default_concurrency_limit,).launch(
                inbrowser=self.gr_is_inbrowser,
                quiet=self.gr_is_quiet,
                server_name=self.gr_server_name,
                server_port=self.gr_transcriber_webui_port,
                share=self.gr_is_share,
            )


if __name__ == "__main__":
    webui = TranscriberWebUI()
    webui()
