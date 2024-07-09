from pathlib import Path
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Dict

import mimetypes
from funasr import AutoModel

import config as cfg
from i18n import I18nAuto


class AudioTranscriber(object):

    def __init__(self):
        self.i18n = I18nAuto()
        self.proc_count = 0
        self.success_count = 0

        self.device = cfg.device
        self.funasr_large_model_path = cfg.funasr_large_model_path
        self.funasr_vad_model_path = cfg.funasr_vad_model_path
        self.funasr_punc_model_path = cfg.funasr_punc_model_path

        self.funasr_model = AutoModel(
            model=self.funasr_large_model_path,
            vad_model=self.funasr_vad_model_path,
            punc_model=self.funasr_punc_model_path,
            device=self.device,
            sentence_timestamp=True
        )

    def __format_time(
        self,
        time: int
    ) -> str:
        h, m = divmod(time, 3600000)
        m, s = divmod(m, 60000)
        s, ms = divmod(s, 1000)

        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def open_transcriber(
        self,
        input: Optional[tuple[str]],
        output: str
    ) -> Generator[Tuple[str, Dict[str, str | bool]], None, None]:
        if input is None:
            error_msg = self.i18n("请上传需要转写的音频。")
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
            sub_path = output_path / audio_name
            sub_path.mkdir(parents=True, exist_ok=True)
            output_subtitle_name_ext = file.with_suffix(".srt").name
            output_subtitle_path = str(sub_path / output_subtitle_name_ext)

            self.proc_count += 1

            transcribing_msg = self.i18n(f"转写中：{audio_path}。")
            yield transcribing_msg, {"__type__": "update", "visible": False}
            print(transcribing_msg)

            with open(output_subtitle_path, "w", encoding="utf-8") as file:
                sentence_info = self.funasr_model.generate(audio_path)[0]["sentence_info"]
                for index, sentence in enumerate(sentence_info, start=1):
                    start_time = self.__format_time(sentence["start"])
                    end_time = self.__format_time(sentence["end"])
                    text = sentence["text"]

                    file.write(f"{index}\n{start_time} --> {end_time}\n{text}\n\n")

                self.success_count += 1

        done_msg = self.i18n(f"转写完毕，检测到总共有 {self.proc_count} 个文件，最终成功转写 {self.success_count} 个文件。")
        yield done_msg, {"__type__": "update", "visible": True}
        print(done_msg)
