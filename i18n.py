import yaml
import locale
import os


def load_language_list(language):
    with open(f"src/i18n/locale/{language}.yaml", "r", encoding="utf-8") as file:
        language_list = yaml.safe_load(file)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[0]
        if not os.path.exists(f"./i18n/locale/{language}.yaml"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
