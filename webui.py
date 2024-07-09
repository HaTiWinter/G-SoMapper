import sys
from pathlib import Path
current_path = Path(__file__).parent
sys.path.insert(0, str(current_path))

import gradio as gr

import config as cfg
from i18n import I18nAuto
from slicer import AudioSlicer
from normalizer import AudioNormalizer
from remover import AudioRemover
from transcriber import AudioTranscriber
from packer import open_packer


class WebUI(object):

    def __init__(self):
        self.local_url = cfg.local_url
        self.os_name = cfg.os_name
        self.python_ver = cfg.python_ver
        self.device = cfg.device
        self.gpu_name = cfg.gpu_name

        self.gr_title = cfg.gr_title
        self.gr_theme = cfg.gr_theme
        self.gr_max_size = cfg.gr_max_size
        self.gr_default_concurrency_limit = cfg.gr_default_concurrency_limit
        self.gr_is_inbrowser = cfg.gr_is_inbrowser
        self.gr_is_quiet = cfg.gr_is_quiet
        self.gr_server_name = cfg.gr_server_name
        self.gr_webui_port = cfg.gr_webui_port
        self.gr_is_share = cfg.gr_is_share

        self.i18n = I18nAuto()
        self.slicer = AudioSlicer()
        self.norm = AudioNormalizer()
        self.remover = AudioRemover()
        self.tran = AudioTranscriber()

        print(self.i18n(f"Running on local URL: {self.local_url}"))
        print(self.os_name, self.python_ver, self.device, self.gpu_name)

    def webui(self):
        with gr.Blocks(title=self.gr_title, theme=self.gr_theme) as app:
            gr.Markdown(self.i18n("# G-SoMapper WebUI"))
            gr.Markdown(self.i18n("**注意：UVR GUI 测试版和 Aegisub 的安装包在根目录的 packages 文件夹下。**"))
            gr.Markdown(self.i18n("##### 请按步骤开始构建您的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 训练数据集："))
            with gr.Tabs():
                with gr.TabItem(self.i18n("1. 准备音频")):
                    with gr.TabItem(self.i18n("1.1. 切分音频")):
                        gr.Markdown(self.i18n("##### 切分过长的视频或音频，输出 WAV 格式的 16 位 44100 Hz 单声道音频，防止后续 UVR 和 ASR 爆内存或爆显存。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("请上传需要切分的视频或音频。"))
                                        slicer_input_path = gr.File(label=self.i18n("上传文件"), type="filepath", file_count="multiple", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("经过切分处理后，音频文件的输出目录。"))
                                        slicer_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("每段音频切片的持续时间。此值越小，生成的音频切片就越多。如果过滤音频出现内存溢出或显存溢出错误，请将此值设置得更小。**默认值为 600（秒）。**"))
                                        slicer_seg_dur = gr.Number(label=self.i18n("片段持续时间（秒）"), value=600, step=60, precision=0, interactive=True)
                            with gr.Column():
                                slicer_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_slicer_btn = gr.Button(self.i18n("开始切分"), variant="primary", visible=True)
                        open_slicer_btn.click(self.slicer.open_slicer, [slicer_input_path, slicer_output_path, slicer_seg_dur], [slicer_info, open_slicer_btn])
                    with gr.TabItem(self.i18n("1.2. 过滤音频")):
                        gr.Markdown(self.i18n("##### 过滤无关音频数据，优化音频质量。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    gr.Markdown(self.i18n("点击“开始过滤”按钮将打开 UVR GUI。经测试，以下 UVR 模型组合效果最好："))
                                    gr.Markdown(self.i18n("**1. 提取人声：MDX23C-InstVoc HQ 或 BS-Roformer-ViperX-1297。**"))
                                    gr.Markdown(self.i18n("**2. 去除混响：Reverb HQ。**"))
                                    gr.Markdown(self.i18n("**3. 去除回声：UVR-De-Echo-Aggressive。**"))
                                    gr.Markdown(self.i18n("**4. 去除噪音：UVR-DeNoise。**"))
                                    gr.Markdown(self.i18n("**按步骤调用模型即可获得最佳效果。**"))
                                    gr.Markdown(self.i18n("注意："))
                                    gr.Markdown(self.i18n("1. BS-Roformer 系列模型的速度和质量是 MDX23C 系列模型的 2 倍，显存仅占用 MDX23C 系列模型的一半，但是 BS-Roformer 系列模型的使用需要安装 UVR GUI 测试版。暂时仅支持 Windows 平台。 **如果条件允许，请优先考虑使用 BS-Roformer 系列模型。** 推荐使用版本号更新的 BS-Roformer 模型，目前最新的是 BS-Roformer-ViperX-1297。"))
                                    gr.Markdown(self.i18n("2. UVR-De-Echo 系列模型有三个档位，Normal、Aggressive，对应两种不同的回声强度，回声越大，选择的模型挡位也应当越高。Dereverb 挡位在 Aggressive 的基础上额外提供了去除混响功能，**但是 Reverb HQ 已经将混响去除，故不再重复此操作。**"))
                                    gr.Markdown(self.i18n("3. UVR-DeNoise 系列模型 **只推荐使用不带有 Lite 后缀的版本** ，即 UVR-DeNoise。"))
                                    gr.Markdown(self.i18n("4. Reverb HQ 模型的效果比 UVR-De-Echo-Dereverb 好"))
                                    gr.Markdown(self.i18n("5. **[更详细的 UVR 使用教程](https://www.bilibili.com/read/cv27499700)（bilibili）**"))
                            with gr.Column():
                                remover_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_remover_btn = gr.Button(self.i18n("开始过滤"), variant="primary", visible=True)
                        open_remover_btn.click(self.remover.open_remover, [], [remover_info, open_remover_btn])
                    with gr.TabItem(self.i18n("1.3. 归一化音频")):
                        gr.Markdown(self.i18n("##### 均衡音频响度，优化音频质量；输出 WAV 格式的 16 位 32000 Hz 单声道音频，方便后续进一步处理。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("请上传需要归一化的音频。"))
                                        norm_input_path = gr.File(label=self.i18n("上传音频"), type="filepath", file_count="multiple", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("经过归一化处理后，音频文件的输出目录。"))
                                        norm_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("执行 ITU-R BS.1770-4 标准。该值越大，响度越大。**默认值为 -16.0。**"))
                                        target_loud = gr.Slider(label=self.i18n("目标响度（分贝）"), minimum=-36.0, maximum=-6.0, value=-16.0, step=0.1, interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("该值越大，响度越大。**默认值为 -1.0。**"))
                                        max_peak = gr.Slider(label=self.i18n("最大振幅（分贝）"), minimum=-12.0, maximum=-0.1, value=-1.0, step=0.1, interactive=True)
                            with gr.Column():
                                norm_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_norm_btn = gr.Button(self.i18n("开始归一化"), variant="primary", visible=True)
                        open_norm_btn.click(self.norm.open_normalizer, [norm_input_path, norm_output_path, target_loud, max_peak], [norm_info, open_norm_btn])
                with gr.TabItem(self.i18n("2. 准备标注")):
                    with gr.TabItem(self.i18n("2.1. 生成标注")):
                        gr.Markdown(self.i18n("##### 生成准确率较高的字幕，需要后期手动校对。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("请上传需要转写的音频。"))
                                        tran_input_path = gr.File(label=self.i18n("上传音频"), type="filepath", file_count="multiple", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("字幕文件的输出目录。"))
                                        tran_output_path = gr.Textbox(label=self.i18n("输出目录"), interactive=True)
                            with gr.Column():
                                tran_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_tran_btn = gr.Button(self.i18n("开始生成"), variant="primary", visible=True)
                        open_tran_btn.click(self.tran.open_transcriber, [tran_input_path, tran_output_path], [tran_info, open_tran_btn])
                    with gr.TabItem(self.i18n("2.2. 合并标注")):
                        gr.Markdown(self.i18n("##### 合并字幕，方便在 GUI 中手动校对。"))
                    with gr.TabItem(self.i18n("2.3. 校对标注")):
                        gr.Markdown(self.i18n("##### 在 GUI 中手动校对 ASR 生成的字幕，最终形成精确字幕。"))
                with gr.TabItem(self.i18n("3. 准备数据")):
                    with gr.TabItem(self.i18n("3.1 切分数据")):
                        gr.Markdown(self.i18n("##### 根据校对后的 SRT 字幕切分 WAV 音频，并且生成 LIST 标注。"))
                    with gr.TabItem(self.i18n("3.2 合并数据")):
                        gr.Markdown(self.i18n("##### 根据 LIST 标注和指定的合并节点合并 WAV 音频，并且生成新的 LIST 标注。"))
                    with gr.TabItem(self.i18n("3.3 打包数据")):
                        gr.Markdown(self.i18n("##### 将合并后的数据打包成适用于 GPT-SoVITS 的训练数据集，可以直接投入训练。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("需打包的音频文件和同名的字幕文件所在目录的路径。**默认值为 output_revised。**"))
                                        packer_input_path = gr.Textbox(label=self.i18n("Input: 输入目录"), value="output_revised", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("打包后的训练数据集的输出目录。**默认值为 dataset。**"))
                                        packer_output_path = gr.Textbox(label=self.i18n("Output: 输出目录"), value="dataset", interactive=True)
                            with gr.Column():
                                packer_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_packer_btn = gr.Button(self.i18n("开始打包"), variant="primary", visible=True)
                        open_packer_btn.click(open_packer, [packer_input_path, packer_output_path], [packer_info, open_packer_btn])
                with gr.TabItem(self.i18n("4. 参考音频")):
                    with gr.TabItem(self.i18n("4.1. 情感识别")):
                        gr.Markdown(self.i18n("##### 施工中，请稍等……"))
            app.queue(max_size=self.gr_max_size, default_concurrency_limit=self.gr_default_concurrency_limit).launch(
                inbrowser=self.gr_is_inbrowser,
                quiet=self.gr_is_quiet,
                server_name=self.gr_server_name,
                server_port=self.gr_webui_port,
                share=self.gr_is_share
            )

if __name__ == "__main__":
    main = WebUI()
    main.webui()
