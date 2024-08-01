import sys
from pathlib import Path


current_path: Path = Path(__file__).parent
current_path_str: str = str(current_path)
sys.path.insert(0, str(current_path))

import gradio as gr
from gradio.themes.default import Default

from config import Config
from i18n import I18nAuto
from label.transcriber import AudioTranscriber


class TranscriberWebUI(object):

    def __init__(self):
        self.cfg = Config()
        self.i18n = I18nAuto()
        self.tran = AudioTranscriber()

        self.gr_transcriber_title: str = self.cfg.gr_transcriber_title
        self.gr_theme: Default = self.cfg.gr_theme
        self.gr_max_size: int = self.cfg.gr_max_size
        self.gr_default_concurrency_limit: int = self.cfg.gr_default_concurrency_limit
        self.gr_is_inbrowser: bool = self.cfg.gr_is_inbrowser
        self.gr_is_quiet: bool = self.cfg.gr_is_quiet
        self.gr_is_share: bool = self.cfg.gr_is_share
        self.gr_server_name: str = self.cfg.gr_server_name
        self.gr_transcriber_webui_port: int = self.cfg.gr_transcriber_webui_port

        self.transcriber_local_url: str = self.cfg.transcriber_local_url

        print(self.i18n(f"Transcriber running on local URL: {self.transcriber_local_url}"))

    def webui(self) -> None:
        with gr.Blocks(title=self.gr_transcriber_title, theme=self.gr_theme) as app:
            gr.Markdown(self.i18n("# Transcriber - G-SoMapper WebUI"))
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
                        tran_info = gr.Textbox(label=self.i18n("输出"), interactive=False)
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
    tran = TranscriberWebUI()
    tran.webui()
