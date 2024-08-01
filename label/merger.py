from pathlib import Path
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Dict

import librosa
import mimetypes
import numpy as np
import soundfile as sf

from config import Config
from i18n import I18nAuto


class AudioMerger(object):

    def __init__(self) -> None:
        self.cfg = Config()
        self.i18n = I18nAuto()

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
    ) -> Generator[Tuple[str, Dict[str, str | bool]], None, None]:
        if file_input_a is None or file_input_b is None:
            error_msg: str = self.i18n("请上传需要合并的同名的音频和字幕。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        file_list_a = (Path(file_path) for file_path in file_input_a)
        file_list_b = (Path(file_path) for file_path in file_input_b)
        output_path = Path(output)

        self.proc_count = 0
        self.success_count = 0
        self.audio_path_list = []
        self.subtitle_path_list = []
        self.audio_name_list = []
        self.subtitle_name_list = []
        self.audio_data_list =  []

        for file in file_list_a:
            type = mimetypes.guess_type(file)[0]
            if (type is None or not (type == "audio/wav")):
                continue_msg = self.i18n(f"跳过：{file}")
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            audio_path = file
            self.audio_path_list.append(audio_path)
        for file in file_list_b:
            type = mimetypes.guess_type(file)[0]
            if (type is None or not (type == "application/x-srt")):
                continue_msg = self.i18n(f"跳过：{file}")
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            subtitle_path = file
            self.subtitle_path_list.append(subtitle_path)
        if len(self.audio_path_list) != len(self.subtitle_path_list):
            error_msg = self.i18n("请确保上传的音频数量与字幕数量相匹配。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        for audio in self.audio_path_list:
            audio_name_i = audio.stem
            self.audio_name_list.append(audio_name_i)
        for subtitle in self.subtitle_path_list:
            subtitle_name_i = subtitle.stem
            self.subtitle_name_list.append(subtitle_name_i)
        for i in range(len(self.audio_path_list)):
            if self.audio_name_list[i] == self.subtitle_name_list[i]:
                audio_path = self.audio_path_list[i]
                subtitle_path = self.subtitle_path_list[i]
                audio_name = audio_path.stem.split("_")[0]
                sub_path = output_path / audio_name
                sub_path.mkdir(parents=True, exist_ok=True)
                file_path = str(file_path)
                audio_name_ext = f"{audio_name}.wav"    
                output_audio_path = str(sub_path / audio_name_ext)
                output_subtitle_name_ext = f"{audio_name}.srt"
                output_subtitle_path = sub_path / output_subtitle_name_ext
                duration = librosa.get_duration(path=audio_path)
                end_time = self._format_time(int(duration * 1000))
                audio_data = librosa.load(audio_path)[0]
                self.audio_data_list.append(audio_data)
                self.proc_count += 1
                with open(str(subtitle_path), "r", encoding="utf-8") as f:
                    subtitle_data = f.readlines()
                    text = subtitle_data[2]
                with open (str(output_subtitle_path), "w", encoding="utf-8") as f:
                    f.write(f"1\n00:00:00,000 --> {end_time}\n{text}\n\n")
        merged_audio_data = np.concatenate(self.audio_data_list)
        sf.write(
            output_audio_path,
            merged_audio_data,
            32000,
            subtype="PCM_16",
            endian="LITTLE",
            format="WAV"
        )
