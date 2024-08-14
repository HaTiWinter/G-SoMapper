from pathlib import Path
from typing import Optional
from typing import Generator

from i18n import I18nAuto

class Packer(object):

    def __init__(self) -> None:
        self.i18n = I18nAuto()

    def __call__(
        self,
        file_input_a: Optional[tuple[str]],
        file_input_b: Optional[tuple[str]],
        output: str
    ) -> Generator[tuple[str, dict[str, str | bool]], None, None]:
        if file_input_a is None or file_input_b is None:
            error_msg = self.i18n("请上传需要合并的同名的音频和字幕。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        file_list_a = (Path(file_path) for file_path in file_input_a)
        file_list_b = (Path(file_path) for file_path in file_input_b)
        output_path = Path(output)

        proc_count = 0
        success_count = 0
        audio_path_list = []
        subtitle_path_list = []

        for file in file_list_a:
            type = file.suffix
            if type == '' or not type == ".wav":
                continue_msg = f"passed：{file}"
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            audio_path = file
            audio_path_list.append(audio_path)
        for file in file_list_b:
            type = file.suffix
            if type == '' or not type == ".srt":
                continue_msg = f"passed：{file}"
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            subtitle_path = file
            subtitle_path_list.append(subtitle_path)
        audio_path_list_len = len(audio_path_list)
        subtitle_path_list_len = len(subtitle_path_list)
        if audio_path_list_len != subtitle_path_list_len:
            error_msg = "请确保上传的音频数量与字幕数量相匹配。"
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        proc_count = audio_path_list_len

        merging_msg = f"打包中：检测到总共有 {proc_count} 组文件"
        print(merging_msg)
        yield merging_msg, {"__type__": "update", "visible": False}

