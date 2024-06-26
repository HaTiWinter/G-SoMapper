from converter import open_converter
from i18n import I18nAuto
from merger import open_merger
from normalizer import open_normalizer
from packer import open_packer
from remover import open_remover
from slicer import open_slicer
from transcriber import open_transcriber
i18n = I18nAuto()
import config as cfg
import gradio as gr


def webui():
    with gr.Blocks(title=cfg.gr_title, theme=cfg.gr_theme) as app:
        gr.Markdown(i18n("# G-SoMapper WebUI"))
        gr.Markdown(i18n("##### 请按步骤开始构建您的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 训练数据集："))
        with gr.Tabs():
            with gr.TabItem(i18n("1. 准备音频")):
                with gr.TabItem(i18n("1.1. 转换格式")):
                    gr.Markdown(i18n("##### 将视频或音频转换为 WAV 格式的 24 位单声道音频，防止后续处理出错。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("需转换的音频文件或所在目录的路径。**默认值为 input。**"))
                                    conv_input_path = gr.Textbox(label=i18n("Input: 输入路径"), value="input", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("转换后的音频的输出目录。**默认值为 output/converted。**"))
                                    conv_output_path = gr.Textbox(label=i18n("Output: 输出目录"), value="output/converted", interactive=True)
                        with gr.Column():
                            conv_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_conv_btn = gr.Button(i18n("开始转换"), variant="primary", visible=True)
                    open_conv_btn.click(open_converter, [conv_input_path, conv_output_path], [conv_info, open_conv_btn])
                with gr.TabItem(i18n("1.2. 切分音频")):
                    gr.Markdown(i18n("##### 缩短长音频，防止后续 UVR 爆内存或爆显存。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("需切分的音频文件或所在目录的路径。**默认值为 output/converted。**"))
                                    slicer_input_path = gr.Textbox(label=i18n("Input: 输入路径"), value="output/converted", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("切分后的音频的输出目录。**默认值为 output/sliced。**"))
                                    slicer_output_path = gr.Textbox(label=i18n("Output: 输出目录"), value="output/sliced", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("RMS 阈值。所有 RMS 值低于此阈值的区域将视为静音。如果音频有噪音，请增加此值。**默认值为 -36.0。**"))
                                    slicer_threshold = gr.Number(label=i18n("Threshold (dB): 静音阈值"), value=-36.0, step=0.1, precision=1, interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("切片后的音频片段的最小长度。**默认值为 60000。**"))
                                    slicer_min_len = gr.Number(label=i18n("Min Length (ms): 最小长度"), value=60000, step=1000, precision=0, interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("要切片的静音部分的最小长度。如果音频包含短暂的中断，请将此值设置得更小。此值越小，生成的切片音频片段就越多。此值必须小于 Min Length 且大于 Hop Size。**默认值为 300。**"))
                                    slicer_min_interval = gr.Number(label=i18n("Min Interval (ms): 最小间距"), value=300, step=10, precision=0, interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("每个 RMS 帧的长度。增加此值将提高切片的精度，但会降低处理速度。**默认值为 10。**"))
                                    slicer_hop_size = gr.Number(label=i18n("Hop Size (ms): 跳跃步长"), value=10, step=1, precision=0, interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("在切片音频周围的最大静音长度。设置此值不意味着切片音频中的静音部分具有完全给定的长度。该算法将搜索要切片的最佳位置。**默认值为 1000。**"))
                                    slicer_max_silence_len = gr.Number(label=i18n("Max Silence Length (ms): 最大静音长度"), value=1000, step=100, precision=0, interactive=True)
                        with gr.Column():
                            slicer_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_slicer_btn = gr.Button(i18n("开始切分"), variant="primary", visible=True)
                    open_slicer_btn.click(open_slicer, [slicer_input_path, slicer_output_path, slicer_threshold, slicer_min_len, slicer_min_interval, slicer_hop_size, slicer_max_silence_len], [slicer_info, open_slicer_btn])
                with gr.TabItem(i18n("1.3. 过滤音频")):
                    gr.Markdown(i18n("##### 过滤无关音频数据，优化音频质量。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown(i18n("打开过滤"))
                        with gr.Column():
                            remover_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_remover_btn = gr.Button(i18n("开始过滤"), variant="primary", visible=True)
                    open_remover_btn.click(open_remover, [], [remover_info, open_remover_btn])
                with gr.TabItem(i18n("1.4. 均衡响度")):
                    gr.Markdown(i18n("##### 均衡音频响度，优化音频质量。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("需均衡响度的音频文件或所在目录的路径。**默认值为 output_speech。**"))
                                    norm_input_path = gr.Textbox(label=i18n("Input: 输入路径"), value="output_speech", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("响度均衡后音频的输出目录。**默认值为 output_normalized。**"))
                                    norm_output_path = gr.Textbox(label=i18n("Output: 输出目录"), value="output_normalized", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("目标响度。该值越大，响度越大。**默认值为 -16.0。**"))
                                    target_lufs = gr.Slider(label=i18n("Target LUFS (dB): 目标响度"), minimum=-36, maximum=-6, value=-16.0, step=0.1, interactive=True)
                        with gr.Column():
                            norm_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_norm_btn = gr.Button(i18n("开始处理"), variant="primary", visible=True)
                    open_norm_btn.click(open_normalizer, [norm_input_path, norm_output_path, target_lufs], [norm_info, open_norm_btn])
                with gr.TabItem(i18n("1.5. 合并音频")):
                    gr.Markdown(i18n("##### 将多个短音频片段合并成单个长音频文件，方便后续进一步处理。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("需合并的音频片段所在目录的路径。**默认值为 output_normalized。**"))
                                    merger_input_path = gr.Textbox(label=i18n("输入文件夹路径"), value="output_normalized", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    gr.Markdown(i18n("合并后音频的输出目录。**默认值为 output_merged。**"))
                                    merger_output_path = gr.Textbox(label=i18n("输出文件夹路径"), value="output_merged", interactive=True)
                        with gr.Column():
                            merger_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_merger_btn = gr.Button(i18n("开始合并"), variant="primary", visible=True)
                    open_merger_btn.click(open_merger, [merger_input_path, merger_output_path], [merger_info, open_merger_btn])
            with gr.TabItem(i18n("2. 准备标注")):
                with gr.TabItem(i18n("2.1. 生成字幕")):
                    gr.Markdown(i18n("##### 生成准确率较高的字幕，之后可以在 GUI 中手动校对。"))
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                with gr.Row():
                                    asr_inp_dir = gr.Textbox(label=i18n("输入文件夹路径"), value="output_merged", interactive=True)
                                with gr.Row():
                                    asr_opt_dir = gr.Textbox(label=i18n("输出文件夹路径"), value="output/asr_opt", interactive=True)
                            with gr.Group():
                                with gr.Row():
                                    asr_model = gr.Dropdown(label=i18n("ASR 模型"), choices=list(cfg.asr_dict.keys()), interactive=True, value="达摩 ASR（中文）")
                            with gr.Group():
                                with gr.Row():
                                    asr_size = gr.Dropdown(label=i18n("ASR 模型尺寸"), choices=["large"], interactive=True, value="large")
                            with gr.Group():
                                with gr.Row():
                                    asr_lang = gr.Dropdown(label=i18n("ASR 语言设置"), choices=["zh"], interactive=True, value="zh")
                        with gr.Column():
                            tran_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                            open_tran_btn = gr.Button(i18n("开始生成"), variant="primary", visible=True)
                    def change_lang_choices(key):
                        return {"__type__": "update", "choices": cfg.asr_dict[key]['lang'], "value": cfg.asr_dict[key]['lang'][0]}
                    def change_size_choices(key):
                        return {"__type__": "update", "choices": cfg.asr_dict[key]['size']}

                    asr_model.change(change_lang_choices, [asr_model], [asr_lang])
                    asr_model.change(change_size_choices, [asr_model], [asr_size])
                    open_tran_btn.click(open_transcriber, [], [tran_info, open_tran_btn])
                with gr.TabItem(i18n("2.2. 校对字幕")):
                    gr.Markdown(i18n("##### 在 GUI 中手动校对 ASR 生成的字幕，最终形成精确字幕。"))
            with gr.TabItem(i18n("3. 打包数据")):
                gr.Markdown(i18n("##### 打包成适用于 GPT-SoVITS 的训练数据集，可以直接投入训练。"))
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown(i18n("需打包的音频文件和同名的字幕文件所在目录的路径。**默认值为 output_revised。**"))
                                packer_input_path = gr.Textbox(label=i18n("Input: 输入目录"), value="output_revised", interactive=True)
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown(i18n("打包后的训练数据集的输出目录。**默认值为 dataset。**"))
                                packer_output_path = gr.Textbox(label=i18n("Output: 输出目录"), value="dataset", interactive=True)
                    with gr.Column():
                        packer_info = gr.Textbox(label=i18n("进程输出信息"), interactive=False)
                        open_packer_btn = gr.Button(i18n("开始打包"), variant="primary", visible=True)
                open_packer_btn.click(open_packer, [packer_input_path, packer_output_path], [packer_info, open_packer_btn])
        app.queue(max_size=cfg.gr_max_size).launch(
            inbrowser=cfg.gr_is_inbrowser,
            quiet=cfg.gr_is_quiet,
            server_name=cfg.gr_server_name,
            server_port=cfg.gr_webui_port_main,
            share=cfg.gr_is_share,
        )

if __name__ == "__main__":
    print(f"Running on local URL: {cfg.local_url}")
    webui()
