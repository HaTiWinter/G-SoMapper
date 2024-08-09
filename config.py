import os
import sys

import gradio as gr


class Config:

    def __init__(self) -> None:
        self.gr_theme = gr.themes.Default()
        self.gr_max_size = int(os.environ.get("max_size", 1024))
        self.gr_default_concurrency_limit = int(os.environ.get("default_concurrency_limit", 512))
        self.gr_is_inbrowser = False if os.environ.get("is_inbrowser", "True").lower() == "false" else True
        self.gr_is_quiet = True if os.environ.get("is_quiet", "False").lower() == "false" else False
        self.gr_is_share = True if os.environ.get("is_share", "False").lower() == "true" else False
        self.gr_server_name = "0.0.0.0"

        self.os_name = sys.platform
