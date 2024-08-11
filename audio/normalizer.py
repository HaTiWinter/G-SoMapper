from pathlib import Path
from typing import Generator
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from pyloudnorm import Meter

from i18n import I18nAuto


class Normalizer(object):

    def __init__(self) -> None:
        self.i18n = I18nAuto()

    def _normalize_loudness(
        self,
        audio_data: np.ndarray,
        input_loud: float,
        target_loud: float,
        target_max_peak: float
    ) -> np.ndarray:
        audio_max_peak = np.max(np.abs(audio_data))
        target_max_peak = np.power(10.0, target_max_peak / 20.0)

        delta_loud = target_loud - input_loud
        gain = np.power(10.0, delta_loud / 20.0)
        projected_peak = audio_max_peak * gain

        total_gain = gain
        if projected_peak >= target_max_peak:
            reduction_factor = target_max_peak / projected_peak
            total_gain = gain * reduction_factor
        normalized_audio_data = total_gain * audio_data

        return normalized_audio_data

    def __call__(
        self,
        input: Optional[tuple[str]],
        output: str,
        target_loud: float,
        max_peak: float
    ) -> Generator[tuple[str, dict[str, str | bool]], None, None]:
        if input is None:
            error_msg = self.i18n("请上传需要归一化的音频。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        file_list = (Path(file_path) for file_path in input)
        output_path = Path(output)

        self.proc_count = 0
        self.success_count = 0
        self.audio_path_list = []
        self.output_audio_path_list = []
        self.buffer = {}

        for file in file_list:
            file_path = str(file)
            type = file.suffix
            if type == '' or not type == ".wav":
                continue_msg = self.i18n(f"跳过：{file_path}。")
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            self.audio_path_list.append(str(file_path))
            self.proc_count += 1

            audio_name = file.stem.split("_")[0]
            audio_name_ext = file.name
            sub_path = output_path / f"{audio_name}_normalized"
            sub_path.mkdir(parents=True, exist_ok=True)
            output_audio_path = sub_path / audio_name_ext
            self.output_audio_path_list.append(str(output_audio_path))

        normalizing_msg = self.i18n(f"归一化中：检测到总共有 {self.proc_count} 个文件")
        print(normalizing_msg)
        yield normalizing_msg, {"__type__": "update", "visible": False}
        audio_path_list_len = len(self.audio_path_list)
        for i in range(audio_path_list_len):
            audio_path = self.audio_path_list[i]
            output_audio_path = self.output_audio_path_list[i]

            audio_data, sr = librosa.load(audio_path, sr=None)
            audio_duration_s = librosa.get_duration(y=audio_data, sr=sr)
            if audio_duration_s == 0:
                error_msg = self.i18n(f"归一化失败：请确保输入音频不为空 -> {audio_path}")
                print(error_msg)
                yield error_msg, {"__type__": "update", "visible": False}
                continue

            self.buffer.setdefault(audio_path, {"audio_data": np.zeros(0), "sample_rate": 0.0, "output_path": ''})
            self.buffer[audio_path]["audio_data"] += audio_data
            self.buffer[audio_path]["sample_rate"] += sr
            self.buffer[audio_path]["output_path"] += output_audio_path
        for key, value in self.buffer.items():
            audio_data = value["audio_data"]
            sample_rate = value["sample_rate"]
            output_path = value["output_path"]
            origin_loud = Meter(sample_rate).integrated_loudness(audio_data)
            normalized_audio_data = self._normalize_loudness(
                audio_data,
                origin_loud,
                target_loud,
                max_peak
            )
            resampled_audio_data = librosa.resample(normalized_audio_data, orig_sr=sample_rate, target_sr=32000.0)
            sf.write(
                output_path,
                resampled_audio_data,
                32000,
                subtype="PCM_16",
                endian="LITTLE",
                format="WAV"
            )
            self.success_count += 1
        done_msg = self.i18n(f"归一化完毕：最终成功归一化 {self.success_count} 个文件")
        print(done_msg)
        yield done_msg, {"__type__":"update","visible":True}
