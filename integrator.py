import os
import shutil
from pathlib import Path
from argparse import ArgumentParser
from uuid import uuid4

import librosa
import numpy as np
import soundfile as sf
from py3langid.langid import classify
from py3langid.langid import set_languages

set_languages(langs=None)


def unformat(timestamp) -> int:
    h, m, s, ms = map(int, (timestamp[:2], timestamp[3:5], timestamp[6:8], timestamp[9:]))

    return h * 3600000 + m * 60000 + s * 1000 + ms


def srt_split_wav(subtitle_path, audio_path, output_dir):
    with subtitle_path.open('r', encoding="utf-8") as subtitle, output_dir.joinpath("splitted_mapping.list").open('w', encoding="utf-8") as mapping_list:
        subtitle_data = subtitle.read().split("\n\n")
        y, sr = librosa.load(audio_path, sr=None)

        for block in range(len(subtitle_data) - 1):
            lines = subtitle_data[block].split('\n')
            audio_name = f"{lines[0]}.wav"
            text = '\n'.join(lines[2:])
            mapping_list.write(f"{audio_name}|{text}\n")

            start_time, end_time = [unformat(timestamp) for timestamp in lines[1].split(" --> ")]

            start_time_sec = start_time / 1000
            end_time_sec = end_time / 1000

            audio_segment = y[int(start_time_sec * sr):int(end_time_sec * sr)]

            sf.write(
                str(output_dir / audio_name),
                audio_segment,
                sr,
                subtype="PCM_24",
                endian="LITTLE",
                format="WAV"
            )


def mapping_merge_wav(input_dir, mapping_list_path, output_dir):
    with mapping_list_path.open('r', encoding="utf-8") as mapping_list, \
         output_dir.joinpath("merged_mapping.list").open("w", encoding="utf-8") as new_mapping_list:
        lines = list(mapping_list)
        counter = 1
        texts_buffer = []
        audio_paths_buffer = []

        print(f"\n{mapping_list_path}")
        for line in lines:
            audio_path = input_dir / line.split("|")[0]
            text = line.split('|')[1].replace('\n', '')
            if any(char in {'。', '.', '！', '!', '？', '?'} for char in (text[-2::])) or not line:
                new_audio_name = f"{counter}.wav"
                texts_buffer.append(text)
                merged_text = ''.join(texts_buffer)
                new_mapping_list.write(f"{new_audio_name}|{merged_text}\n")

                audio_paths_buffer.append(audio_path)

                y = []
                sr = None
                for audio_path in audio_paths_buffer:
                    y_temp, sr_temp = librosa.load(audio_path, sr=None)
                    if sr is None:
                        sr = sr_temp
                    elif sr != sr_temp:
                        raise ValueError("Sampling rates do not match.")
                    y.append(y_temp)
                merged_audio, sr = np.concatenate(y), sr

                sf.write(
                    str(output_dir / new_audio_name),
                    merged_audio,
                    sr,
                    subtype="PCM_24",
                    endian="LITTLE",
                    format="WAV"
                )

                audio_paths_buffer.clear()
                texts_buffer.clear()
                counter += 1
            else:
                texts_buffer.append(text)
                audio_paths_buffer.append(audio_path)


def list_pack_wav(mapping_list_path, output_dir, speaker):
    new_mapping_list_path = output_dir / "packed_mapping.list"

    with mapping_list_path.open('r', encoding="utf-8") as mapping_list, new_mapping_list_path.open('a', encoding="utf-8") as new_mapping_list:
        lines = list(mapping_list)

        for line in lines:
            text = line.split('|')[1]
            language = classify(text)[0].upper()
            new_audio_file_name = f"{speaker}_{uuid4()}.wav"
            new_mapping_list.write(f"./{output_dir.parts[-2]}/{output_dir.parts[-1]}/{new_audio_file_name}|{speaker}|{language}|{text}")

            audio_path = line.split('|')[0]
            source_audio_path = mapping_list_path.parent / audio_path
            dest_audio_path = output_dir / new_audio_file_name

            y, sr = librosa.load(source_audio_path, sr=None)

            sf.write(
                str(dest_audio_path),
                y,
                sr,
                subtype="PCM_24",
                endian="LITTLE",
                format="WAV"
            )


class TempDir:
    def __init__(self, temp_path):
        self.path = temp_path

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)


def srt_pack_wav(input_path, output_path, speaker, temp_path):
    temp_path = Path(os.environ.get("TEMP", "temp"))
    splited_path = temp_path / "splitted"
    merged_path = temp_path / "merged"

    for subtitle_path in input_path.rglob("*.srt"):
        output_dir = splited_path / subtitle_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = subtitle_path.with_suffix(".wav")

        srt_split_wav(subtitle_path, audio_path, output_dir)

    for mapping_list_path in splited_path.rglob("*.list"):
        input_dir = mapping_list_path.parent
        output_dir = merged_path / input_dir.parts[-1]
        output_dir.mkdir(parents=True, exist_ok=True)

        mapping_merge_wav(input_dir, mapping_list_path, output_dir)

    for mapping_list_path in merged_path.rglob("*.list"):
        output_dir = output_path / speaker
        output_dir.mkdir(parents=True, exist_ok=True)

        list_pack_wav(mapping_list_path, output_dir, speaker)


def main():
    parser = ArgumentParser(description="根据srt字幕和wav音频打包数据集")
    parser.add_argument("input", type=str, help="处理的目录")
    parser.add_argument("output", type=str, help="输出的目录")
    parser.add_argument("speaker", type=str, help="说话人")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    speaker = args.speaker

    temp_path = Path(f"{uuid4()}")
    temp_path.mkdir(parents=True, exist_ok=True)

    with TempDir(temp_path):
        srt_pack_wav(input_path, output_path, speaker, temp_path)


if __name__ == '__main__':
    main()
