import sys
from pathlib import Path
current_path = Path(__file__).parent
sys.path.insert(0, str(current_path))

import gradio as gr

import config as cfg
from i18n import I18nAuto
from slicer import AudioSlicer
from merger import AudioMerger
from packer import open_packer
from remover import open_remover
from transcriber import open_transcriber



class WebUI(object):

    def __init__(self):
        self.i18n = I18nAuto()
        self.slicer = AudioSlicer()
        self.merger = AudioMerger()

    def webui(self):
        with gr.Blocks(title=cfg.gr_title, theme=cfg.gr_theme) as app:
            gr.Markdown(self.i18n("# G-SoMapper WebUI"))
            gr.Markdown(self.i18n("**注意：UVR GUI 测试版和 Aegisub 的安装包在根目录的 packages 文件夹下。**"))
            gr.Markdown(self.i18n("##### 请按步骤开始构建您的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 训练数据集："))
            with gr.Tabs():
                with gr.TabItem(self.i18n("1. 准备音频")):
                    with gr.TabItem(self.i18n("1.1. 切分音频")):
                        gr.Markdown(self.i18n("##### 切分过长的视频或音频文件，输出 WAV 格式的 16 位 32000 Hz 单声道音频，防止后续 UVR 爆内存或爆显存。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("输入路径。支持输入文件路径和目录路径。**默认值为 input。**"))
                                        slicer_input_path = gr.Textbox(label=self.i18n("输入路径"), value="output/converted", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("输出目录。经过切分处理后，音频文件的输出目录。**默认值为 output_sliced。**"))
                                        slicer_output_path = gr.Textbox(label=self.i18n("输出目录"), value="output_sliced", interactive=True)
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
                                    gr.Markdown(self.i18n("**5. 请设置 UVR 输出 WAV 格式的 PCM_16 音频**"))
                                    gr.Markdown(self.i18n("**[更详细的 UVR 使用教程](https://www.bilibili.com/read/cv27499700)（bilibili）**"))
                            with gr.Column():
                                remover_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_remover_btn = gr.Button(self.i18n("开始过滤"), variant="primary", visible=True)
                        open_remover_btn.click(open_remover, [], [remover_info, open_remover_btn])
                    with gr.TabItem(self.i18n("1.3. 合并音频")):
                        gr.Markdown(self.i18n("##### 合并多个短音频片段，并且均衡音频响度，优化音频质量，方便后续进一步处理。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("输入路径。支持输入目录路径。**默认值为 output_speech。**"))
                                        merger_input_path = gr.Textbox(label=self.i18n("输入目录"), value="output_speech", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("输出目录。经过切分处理后，音频文件的输出目录。**默认值为 output_merged。**"))
                                        merger_output_path = gr.Textbox(label=self.i18n("输出目录"), value="output_merged", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("目标响度。该值越大，响度越大。**默认值为 -16.0。**"))
                                        target_lufs = gr.Slider(label=self.i18n("目标响度（分贝）"), minimum=-36, maximum=-6, value=-16.0, step=0.1, interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        gr.Markdown(self.i18n("最大振幅。该值越大，响度越大。**默认值为 -1.0。**"))
                                        max_peak = gr.Slider(label=self.i18n("最大振幅（分贝）"), minimum=-12, maximum=-0.1, value=-1.0, step=0.1, interactive=True)
                            with gr.Column():
                                merger_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_merger_btn = gr.Button(self.i18n("开始合并"), variant="primary", visible=True)
                        open_merger_btn.click(self.merger.open_merger, [merger_input_path, merger_output_path, target_lufs, max_peak], [merger_info, open_merger_btn])
                with gr.TabItem(self.i18n("2. 准备标注")):
                    with gr.TabItem(self.i18n("2.1. 生成标注")):
                        gr.Markdown(self.i18n("##### 生成准确率较高的字幕，之后可以在 GUI 中手动校对。"))
                        with gr.Row():
                            with gr.Column():
                                with gr.Group():
                                    with gr.Row():
                                        asr_inp_dir = gr.Textbox(label=self.i18n("输入文件夹路径"), value="output_merged", interactive=True)
                                    with gr.Row():
                                        asr_opt_dir = gr.Textbox(label=self.i18n("输出文件夹路径"), value="output/asr_opt", interactive=True)
                                with gr.Group():
                                    with gr.Row():
                                        asr_model = gr.Dropdown(label=self.i18n("ASR 模型"), choices=list(cfg.asr_dict.keys()), interactive=True, value="达摩 ASR（中文）")
                                with gr.Group():
                                    with gr.Row():
                                        asr_size = gr.Dropdown(label=self.i18n("ASR 模型尺寸"), choices=["large"], interactive=True, value="large")
                                with gr.Group():
                                    with gr.Row():
                                        asr_lang = gr.Dropdown(label=self.i18n("ASR 语言设置"), choices=["zh"], interactive=True, value="zh")
                            with gr.Column():
                                tran_info = gr.Textbox(label=self.i18n("进程输出信息"), interactive=False)
                                open_tran_btn = gr.Button(self.i18n("开始生成"), variant="primary", visible=True)
                        def change_lang_choices(key):
                            return {"__type__": "update", "choices": cfg.asr_dict[key]['lang'], "value": cfg.asr_dict[key]['lang'][0]}
                        def change_size_choices(key):
                            return {"__type__": "update", "choices": cfg.asr_dict[key]['size']}

                        asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                        asr_model.change(change_size_choices, [asr_model], [asr_size])
                        open_tran_btn.click(open_transcriber, [], [tran_info, open_tran_btn])
                    with gr.TabItem(self.i18n("2.2. 校对标注")):
                        gr.Markdown(self.i18n("##### 在 GUI 中手动校对 ASR 生成的字幕，最终形成精确字幕。"))
                with gr.TabItem(self.i18n("3. 打包数据")):
                    gr.Markdown(self.i18n("##### 打包成适用于 GPT-SoVITS 的训练数据集，可以直接投入训练。"))
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
            app.queue(max_size=cfg.gr_max_size, default_concurrency_limit=cfg.gr_default_concurrency_limit).launch(
                inbrowser=cfg.gr_is_inbrowser,
                quiet=cfg.gr_is_quiet,
                server_name=cfg.gr_server_name,
                server_port=cfg.gr_webui_port_main,
                share=cfg.gr_is_share,
            )

if __name__ == "__main__":
    webui = WebUI()

    print(webui.i18n(f"Running on local URL: {cfg.local_url}"))
    print(cfg.os_name, cfg.python_ver, cfg.infer_device, cfg.gpu_name)

    webui.webui()
