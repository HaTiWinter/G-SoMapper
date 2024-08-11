import re
import psutil
from pathlib import Path
from typing import Generator
from typing import Optional

import torch
import librosa
from funasr import AutoModel

from i18n import I18nAuto


class Transcriber(object):

    def __init__(self, lang=None) -> None:
        self.i18n = I18nAuto()
        self.pattern = re.compile(r"<[^>]*>")
        self.ncpu = psutil.cpu_count(logical=False)
        self.device, self.gpu_count = self._get_device()

        self.model_path = "src/funasr/models/SenseVoiceSmall"
        self.vad_model_path = "src/funasr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.punc_model_path = "src/funasr/models/punc_ct-transformer_cn-en-common-vocab471067-large"

        self.funasr_model = AutoModel(
            model=self.model_path,
            vad_model=self.vad_model_path,
            punc_model=self.punc_model_path,
            language=lang,
            max_single_segment_time=30000,
            device=self.device,
            ncpu=self.ncpu,
            ngpu=self.gpu_count,
            trust_remote_code=False,
            use_itn=False
        )

    def _get_device(self) -> tuple[str, int]:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
        else:
            device = "cpu"
            gpu_count = 0
        return device, gpu_count

    def _format_time(self, time: int) -> str:
        h, m = divmod(time, 3600000)
        m, s = divmod(m, 60000)
        s, ms = divmod(s, 1000)

        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def Transcriber(
        self,
        input: Optional[tuple[str]],
        output: str
    ) -> Generator[tuple[str, dict[str, str | bool]], None, None]:
        if input is None:
            error_msg = self.i18n("请上传需要转写的音频。")
            print(error_msg)
            yield error_msg, {"__type__": "update", "visible": True}
            return
        file_list = (Path(file_path) for file_path in input)
        output_path = Path(output)

        self.proc_count = 0
        self.success_count = 0
        self.audio_path_list = []
        self.output_subtitle_path_list = []
        self.audio_end_time_list = []
        self.text_list = []
        self.content_buf = {}

        for file in file_list:
            file_path = str(file)
            type = file.suffix
            if type == '' or not type == ".wav":
                continue_msg = self.i18n(f"跳过：{file_path}")
                print(continue_msg)
                yield continue_msg, {"__type__": "update", "visible": False}
                continue
            self.proc_count += 1

            audio_path = file_path
            audio_base_name = file.stem.split("_")[0]
            sub_path = output_path / f"{audio_base_name}_transcribed"
            sub_path.mkdir(parents=True, exist_ok=True)
            output_subtitle_name_ext = file.with_suffix(".srt").name
            output_subtitle_path = str(sub_path / output_subtitle_name_ext)

            audio_data, sr = librosa.load(audio_path)
            audio_duration_s = librosa.get_duration(y=audio_data, sr=sr)
            audio_duration_ms = int(audio_duration_s * 1000)
            audio_end_time = self._format_time(audio_duration_ms)

            self.audio_path_list.append(audio_path)
            self.output_subtitle_path_list.append(output_subtitle_path)
            self.audio_end_time_list.append(audio_end_time)

        transcribing_msg = self.i18n(f"转写中：检测到总共有 {self.proc_count} 个文件")
        print(transcribing_msg)
        yield transcribing_msg, {"__type__": "update", "visible": False}
        res = self.funasr_model.generate(self.audio_path_list)
        for i in range(len(res)):
            text = re.sub(self.pattern, '', res[i]["text"])
            self.text_list.append(text)
        for i, subtitle_path in enumerate(self.output_subtitle_path_list):
            text = self.text_list[i]
            end_time = self.audio_end_time_list[i]
            subtitle_text = f"1\n00:00:00,000 --> {end_time}\n{text}\n\n"
            if subtitle_path not in self.content_buf:
                self.content_buf[subtitle_path] = ''
            self.content_buf[subtitle_path] += subtitle_text
        if self.content_buf != {}:
            for file_path, content in self.content_buf.items():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.success_count += 1
        done_msg = self.i18n(f"转写完毕：最终成功转写 {self.success_count} 个文件")
        print(done_msg)
        yield done_msg, {"__type__": "update", "visible": True}
