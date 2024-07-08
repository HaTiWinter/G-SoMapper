from pathlib import Path
from typing import Generator
from typing import Tuple
from typing import Dict

import librosa
import mimetypes
import numpy as np
import soundfile as sf
from pyloudnorm import Meter

from i18n import I18nAuto


class AudioNormalizer(object):

    def __init__(self):
        self.i18n = I18nAuto()
        self.proc_count = 0
        self.success_count = 0
        self.audio_data_list = []

        self.sr = 32000
        self.subtype = "PCM_16"
        self.endian = "LITTLE"
        self.format = "WAV"

        self.meter = Meter(self.sr)

    def __normalize_loudness(
        self,
        audio_data: np.ndarray,
        input_loud: float,
        target_loud: float,
        target_max_peak: float
    ) -> np.ndarray:
        delta_loud = target_loud - input_loud

        audio_max_peak = np.max(np.abs(audio_data))
        target_max_peak = np.power(10.0, target_max_peak / 20.0)
        gain = np.power(10.0, delta_loud / 20.0)
    
        projected_peak = audio_max_peak * gain

        if projected_peak >= target_max_peak:
            reduction_factor = target_max_peak / projected_peak
            total_gain = gain * reduction_factor
        else:
            total_gain = gain

        normalized_audio_data = total_gain * audio_data

        return normalized_audio_data

    def open_normalizer(
        self,
        input: tuple[str],
        output: str,
        target_loud: float,
        max_peak: float
    ) -> Generator[Tuple[str, Dict[str, str | bool]], None, None]:
        if input is None:
            error_msg = self.i18n("请上传需要切分的视频或音频。")
            yield error_msg, {"__type__": "update", "visible": True}
            print(error_msg)
            return

        file_list = (Path(file_path) for file_path in input)

        output_path = Path(output)

        self.proc_count = 0
        self.success_count = 0

        for file in file_list:
            file_path = str(file)
            type = mimetypes.guess_type(file)[0]
            if (
                type is None
                or not (type == "audio/wav")
            ):
                continue_msg = self.i18n(f"跳过：{file_path}。")
                yield continue_msg, {"__type__": "update", "visible": False}
                print(continue_msg)
                continue

            audio_path = file_path
            audio_name = file.stem.split("_")[0]
            audio_name_ext = file.name
            sub_path = output_path / audio_name
            sub_path.mkdir(parents=True, exist_ok=True)
            output_audio_path = str(sub_path / audio_name_ext)

            self.proc_count += 1

            normalizing_msg = self.i18n(f"归一化中：{audio_path}。")
            yield normalizing_msg, {"__type__": "update", "visible": False}
            print(normalizing_msg)

            audio_data = librosa.load(audio_path, sr=None)[0]

            normalized_audio_data_len = len(audio_data)
            audio_duration = normalized_audio_data_len / self.sr
            if audio_duration == 0:
                error_msg = self.i18n(f"归一化失败：{audio_path}，请确保输入文件包含有效的音频流。")
                yield error_msg, {"__type__": "update", "visible": False}
                print(error_msg)
                continue

            origin_loud = self.meter.integrated_loudness(audio_data)
            normalized_audio_data = self.__normalize_loudness(
                audio_data,
                origin_loud,
                target_loud,
                max_peak
            )

            sf.write(
                output_audio_path,
                normalized_audio_data,
                self.sr,
                subtype=self.subtype,
                endian=self.endian,
                format=self.format
            )

            self.success_count += 1

        done_msg = self.i18n(f"归一化完毕，检测到总共有 {self.proc_count} 个文件，最终成功归一化 {self.success_count} 个文件。")
        print(done_msg)
        yield done_msg, {"__type__":"update","visible":True}
