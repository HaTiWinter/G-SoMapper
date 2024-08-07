import subprocess as subp
from pathlib import Path
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Dict

import mimetypes
import numpy as np
import soundfile as sf
from audio.__audio_slicer import Slicer

from i18n import I18nAuto


class AudioSlicer(object):

    def __init__(
        self,
        i_threshold: float,
        i_min_length: int,
        i_min_interval: int,
        i_hop_size: int,
        i_max_sil_kept: int
    ) -> None:
        self.i18n = I18nAuto()
        self.proc_count = 0
        self.success_count = 0

        self.sr = 44100
        self.slicer = Slicer(
            sr=self.sr,
            threshold=i_threshold,
            min_length=i_min_length,
            min_interval=i_min_interval,
            hop_size=i_hop_size,
            max_sil_kept=i_max_sil_kept,
        )

    def Slicer(
        self,
        input: Optional[tuple[str]],
        output: str
    ) -> Generator[Tuple[str, Dict[str, str | bool]], None, None]:
        if input is None:
            error_msg = self.i18n("请上传需要切分的视频或音频。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        file_list = (Path(f) for f in input)
        output_path = Path(output)

        self.proc_count = 0
        self.success_count = 0

        for f in file_list:
            file_path = str(f)
            type = mimetypes.guess_type(file_path)[0]
            if (type is None or not type.startswith(("video", "audio"))):
                continue_msg = self.i18n(f"跳过：{file_path}。")
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue

            audio_name = f.stem
            sub_path = output_path / audio_name / "_sliced"
            sub_path.mkdir(parents=True, exist_ok=True)

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-i", file_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-f", "s16le",
                "-ac", "1",
                "-ar", str(self.sr),
                "pipe:1"
            ]
            with open(file_path, "rb") as f:
                with subp.Popen(
                    ffmpeg_cmd,
                    stdin=f,
                    stdout=subp.PIPE,
                    stderr=subp.PIPE
                ) as proc:
                    converting_msg = self.i18n(f"切分中：{file_path}")
                    print(converting_msg)
                    yield converting_msg, {"__type__": "update", "visible": False}
                    self.proc_count += 1

                    proc_out, proc_err = proc.communicate()
                    if proc.returncode != 0:
                        error_msg = self.i18n(f"切分失败：{file_path}，FFmpeg 错误")
                        print(error_msg)
                        print(str(proc_err))
                        yield error_msg, {"__type__": "update", "visible": False}
                        continue

                    audio_data = np.frombuffer(proc_out, dtype=np.int16)
                    self.chunk_count = 1
                    chunks = self.slicer.slice(audio_data)
                    for i, chunk in enumerate(chunks, start=1):
                        output_audio_path = str(sub_path / f"{audio_name}_{i}.wav")
                        sf.write(
                            output_audio_path,
                            chunk,
                            self.sr,
                            subtype="PCM_16",
                            endian="LITTLE",
                            format="WAV"
                        )

                    self.success_count += 1
        done_msg = self.i18n(f"切分完毕：检测到总共有 {self.proc_count} 个文件，最终成功切分 {self.success_count} 个文件")
        print(done_msg)
        yield done_msg, {"__type__": "update", "visible": True}
