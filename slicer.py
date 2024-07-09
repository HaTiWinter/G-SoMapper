import subprocess as subp
from pathlib import Path
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Dict

import mimetypes
import numpy as np
import soundfile as sf

from i18n import I18nAuto


class AudioSlicer(object):

    def __init__(self):
        self.i18n = I18nAuto()
        self.proc_count = 0
        self.success_count = 0
        self.chunk_count = 1

        self.acodec = "pcm_s16le"
        self.f = "s16le"
        self.channels = 1
        self.sr = 44100
        self.subtype = "PCM_16"
        self.endian = "LITTLE"
        self.format = "WAV"

    def __slice(
        self,
        audio_data: np.ndarray,
        audio_data_len: int,
        duration: int
    ) -> Generator[np.ndarray, None, None]:
        total_s = audio_data_len
        s_duration = int(self.sr * duration)
        start_s = 0

        while start_s < total_s:
            end_s = min(start_s + s_duration, total_s)
            chunk = audio_data[start_s:end_s]
            start_s = end_s
            yield chunk

    def open_slicer(
        self,
        input: Optional[tuple[str]],
        output: str,
        duration: int
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
                or not type.startswith(("video", "audio"))
            ):
                continue_msg = self.i18n(f"跳过：{file_path}。")
                yield continue_msg, {"__type__": "update", "visible": False}
                print(continue_msg)
                continue

            audio_name = file.stem
            sub_path = output_path / audio_name
            sub_path.mkdir(parents=True, exist_ok=True)

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-i", file_path,
                "-vn",
                "-acodec", self.acodec,
                "-f", self.f,
                "-ac", str(self.channels),
                "-ar", str(self.sr),
                "pipe:1"
            ]

            with open(file_path, "rb") as file:
                with subp.Popen(
                    ffmpeg_cmd,
                    stdin=file,
                    stdout=subp.PIPE,
                    stderr=subp.PIPE
                ) as ffmpeg_proc:
                    self.proc_count += 1

                    converting_msg = self.i18n(f"切分中：{file_path}。")
                    yield converting_msg, {"__type__": "update", "visible": False}
                    print(converting_msg)

                    proc_out, proc_err = ffmpeg_proc.communicate()
                    if ffmpeg_proc.returncode != 0:
                        error_msg = self.i18n(f"切分失败：{file_path}，FFmpeg 错误。")
                        yield error_msg, {"__type__": "update", "visible": False}
                        print(error_msg)
                        print(str(proc_err))
                        continue

                    audio_data = np.frombuffer(proc_out, dtype=np.int16)
                    audio_data_len = len(audio_data)
                    audio_duration = audio_data_len / self.sr
                    if audio_duration == 0:
                        error_msg = self.i18n(f"切分失败：{file_path}，请确保输入文件包含有效的音频流。")
                        yield error_msg, {"__type__": "update", "visible": False}
                        print(error_msg)
                        continue
                    if audio_duration < duration:
                        error_msg = self.i18n(f'切分失败：输入文件的实际时长 {audio_duration} 秒小于设定的 duration 参数 {duration} 秒。')
                        yield error_msg, {"__type__": "update", "visible": False}
                        print(error_msg)
                        continue

                    self.chunk_count = 1

                    for chunk in self.__slice(audio_data, audio_data_len, duration):
                        output_audio_path = str(sub_path / f"{audio_name}_{self.chunk_count}.wav")
                        sf.write(
                            output_audio_path,
                            chunk,
                            self.sr,
                            subtype=self.subtype,
                            endian=self.endian,
                            format=self.format
                        )
                        self.chunk_count += 1

                    self.success_count += 1

                    success_msg = self.i18n(f"切分成功：{file_path}。")
                    yield success_msg, {"__type__": "update", "visible": False}
                    print(success_msg)

        done_msg = self.i18n(f"切分完毕，检测到总共有 {self.proc_count} 个文件，最终成功切分 {self.success_count} 个文件。")
        yield done_msg, {"__type__": "update", "visible": True}
        print(done_msg)
