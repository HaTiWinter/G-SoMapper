import os
import json
from typing import Optional


class I18nAuto:
    def __init__(self, lang: Optional[str] = None) -> None:
        if lang is None:
            lang = os.environ.get("LANG")
            if lang is not None:
                lang = lang.split(".")[0]
            else:
                raise ValueError("Language not found.")
        if not os.path.exists(f"src/i18n/locale/{lang}.json"):
            lang = "en_US"

        self.lang = lang
        self.lang_map = self._load_lang_list(lang)

    def __call__(self, key) -> str:
        return self.lang_map.get(key, key)

    def __repr__(self) -> str:
        return f"Use Language: {self.lang}"

    def _load_lang_list(self, lang) -> dict[str, str]:
        with open(f"src/i18n/locale/{lang}.json", "r", encoding="utf-8") as f:
            lang_list = json.load(f)
        return lang_list

def main() -> None:
    i18n = I18nAuto()
    print(i18n)

if __name__ == "__main__":
    main()
