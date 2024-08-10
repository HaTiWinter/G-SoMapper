from pathlib import Path
from typing import Generator
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from i18n import I18nAuto


class LabelMerger(object):

    def __init__(self) -> None:
        self.i18n = I18nAuto()

    def _unformat_time(self, timestamp: str) -> int:
        h, m, s, ms = map(int, (timestamp[17:19], timestamp[20:22], timestamp[23:25], timestamp[26:29]))

        return h * 3600000 + m * 60000 + s * 1000 + ms

    def _format_time(self, time: int) -> str:
        h, m = divmod(time, 3600000)
        m, s = divmod(m, 60000)
        s, ms = divmod(s, 1000)

        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def Merger(
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
        index = 0
        start_time = 0
        audio_path_list = []
        subtitle_path_list = []
        audio_data_list = []
        buffer = {}

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

        merging_msg = f"合并中：检测到总共有 {proc_count} 组文件"
        print(merging_msg)
        yield merging_msg, {"__type__": "update", "visible": False}
        for i in range(audio_path_list_len):
            audio_path = audio_path_list[i]
            audio_path_str = str(audio_path)
            subtitle_path = subtitle_path_list[i]
            subtitle_path_str = str(subtitle_path)

            audio_base_name_with_index = audio_path.stem
            subtitle_base_name_with_index = subtitle_path.stem

            audio_base_name = audio_base_name_with_index.split("_")[0] if audio_base_name_with_index == subtitle_base_name_with_index else audio_base_name_with_index.split("_")[0]

            sub_path = output_path / f"{audio_base_name}_merged"
            sub_path.mkdir(parents=True, exist_ok=True)

            audio_name_ext = f"{audio_base_name}.wav"
            output_audio_path = sub_path / audio_name_ext
            output_audio_path_str = str(output_audio_path)

            output_subtitle_name_ext = f"{audio_base_name}.srt"
            output_subtitle_path = sub_path / output_subtitle_name_ext
            output_subtitle_path_str = str(output_subtitle_path)

            audio_data, sr = librosa.load(audio_path_str, sr=None)

            with open(subtitle_path_str, "r", encoding="utf-8") as f:
                subtitle_data = f.readlines()
                index = index + int(subtitle_data[0].strip())
                timestamp = subtitle_data[1].strip()
                subtitle_text = subtitle_data[2].strip()

                end_time = self._unformat_time(timestamp)

            buffer.setdefault(audio_base_name, {
                "audio_data_list": [np.zeros(0)],
                "output_audio_path": '',
                "output_subtitle_path": ''
            }).setdefault(index, {
                "end_time": 0,
                "text": ''
            })
            buffer[audio_base_name]["audio_data_list"].append(audio_data)
            buffer[audio_base_name]["output_audio_path"] = output_audio_path_str
            buffer[audio_base_name]["output_subtitle_path"] = output_subtitle_path_str
            buffer[audio_base_name][index]["end_time"] = end_time
            buffer[audio_base_name][index]["text"] = subtitle_text
        for key, value in buffer.items():
            audio_data_list = value["audio_data_list"]
            output_audio_path_str = value["output_audio_path"]
            output_subtitle_path_str = value["output_subtitle_path"]

            merged_audio_data = np.concatenate(audio_data_list)
            sf.write(
                output_audio_path_str,
                merged_audio_data,
                sr,
                subtype="PCM_16",
                endian="LITTLE",
                format="WAV"
            )

            with open(output_subtitle_path_str, "w", encoding="utf-8") as f:
                for i in range(audio_path_list_len):
                    i += 1
                    end_time = start_time + value[i]["end_time"]
                    text = value[i]["text"]

                    f.write(f"{i}\n{self._format_time(start_time)} --> {self._format_time(end_time)}\n{text}\n\n")

                    start_time = end_time
                    success_count += 1
        done_msg = self.i18n(f"合并完毕：最终成功合并 {success_count} 个文件")
        print(done_msg)
        yield done_msg, {"__type__":"update","visible":True}
