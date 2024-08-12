import subprocess as subp
from pathlib import Path
from typing import Generator
from typing import Optional

import mimetypes
import numpy as np
import soundfile as sf
from librosa.feature.spectral import rms as get_rms

from i18n import I18nAuto


class Slicer(object):

    def __init__(
        self,
        threshold: float = -24.0,
        min_length: int = 5000,
        min_interval: int = 100,
        hop_size: int = 100,
        max_sil_kept: int = 100
    ) -> None:
        self.i18n = I18nAuto()
        self.proc_count = 0
        self.success_count = 0

        self.sr = 48000

        if not min_length >= min_interval >= hop_size:
            raise ValueError("The following condition must be satisfied: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("The following condition must be satisfied: max_sil_kept >= hop_size")

        min_interval_s = self.sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(self.sr * hop_size / 1000)
        self.win_size = min(round(min_interval_s), 4 * self.hop_size)
        self.min_length = round(self.sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_s / self.hop_size)
        self.max_sil_kept = round(self.sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(
            self,
            waveform,
            begin,
            end
        ):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def _slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

    def __call__(
        self,
        input: Optional[tuple[str]],
        output: str
    ) -> Generator[tuple[str, dict[str, str | bool]], None, None]:
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
            sub_path = output_path / f"{audio_name}_sliced"
            sub_path.mkdir(parents=True, exist_ok=True)

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-i", file_path,
                "-vn",
                "-acodec", "pcm_s24le",
                "-f", "s24le",
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
                    chunks = self._slice(audio_data)
                    for i, chunk in enumerate(chunks, start=1):
                        output_audio_path = str(sub_path / f"{audio_name}_{i}.wav")
                        sf.write(
                            output_audio_path,
                            chunk,
                            self.sr,
                            subtype="PCM_24",
                            endian="LITTLE",
                            format="WAV"
                        )

                    self.success_count += 1
        done_msg = self.i18n(f"切分完毕：检测到总共有 {self.proc_count} 个文件，最终成功切分 {self.success_count} 个文件")
        print(done_msg)
        yield done_msg, {"__type__": "update", "visible": True}
