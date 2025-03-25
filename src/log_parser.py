import os
import re
from collections import Counter

import MeCab
from bs4 import BeautifulSoup
from tqdm import tqdm


def safe_decode(text):
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="ignore")
    elif isinstance(text, str):
        try:
            return text.encode("utf-8", errors="ignore").decode("utf-8")
        except:
            return text
    return str(text)


def clean_text(text):
    """
    入力文字列から記号、空白（全角・半角）、制御文字（\u3000 など）を取り除く関数
    """
    return re.sub(r"[^\wぁ-んァ-ヶ一-龥]", "", text)


def check_dialogue(text):
    if "CCB" in text:
        return None
    elif "1d" in text:
        return None
    else:
        return clean_text(text)


def is_dice_command(text):
    """
    ダイスロール系の発言かを判定する関数
    例: '1d100', 'ccb<=55', '2D6+3', 'CCB', 'dx5+1>=10' など
    """
    dice_patterns = [
        r"^[ \[]?(?:\d+)?[dD]\d+",  # 1d100, 2D6 など
        r"^ccb",
        r"^CCB",  # ccbコマンド
        r"^rct",
        r"^choice",  # その他のコマンド
        r"^dx\d+",  # dx式（SW系）
        r"^\s*[a-zA-Z]{2,}[<>=]",  # CCB<=など
    ]
    for pattern in dice_patterns:
        if re.match(pattern, text.strip()):
            return True
    return False


from collections import Counter

import MeCab


def extract_character_features(chat_texts, most_common=5, min_count=1):
    """
    キャラクターのテキスト群から、語尾・一人称・二人称・感嘆詞・名詞・接頭語・定型句などの特徴を抽出。
    出現数の上限（most_common）と、出現回数の下限（min_count）を指定可能。
    """
    mecab = MeCab.Tagger()

    ending_phrases = []
    final_particles = []
    auxiliary_verbs = []
    polite_forms = []
    first_person = []
    second_person = []
    exclamations = []
    common_words = []
    common_nouns = []
    start_phrases = []
    speech_patterns = []

    first_person_list = ["私", "わたし", "僕", "ぼく", "俺", "おれ", "あたし", "ウチ", "うち", "ボク", "自分", "わし"]
    second_person_list = ["あなた", "君", "きみ", "お前", "おまえ", "あんた", "貴様", "そなた"]
    exclamation_candidates = [
        "うわー",
        "うそ",
        "ぎゃー",
        "やった",
        "まじ",
        "すご",
        "わー",
        "ああ",
        "ほんと",
        "ええっ",
        "うーん",
    ]

    for text in chat_texts:
        if is_dice_command(text):
            continue

        node = mecab.parseToNode(text)
        tokens = []
        surfaces = []

        while node:
            surface = node.surface
            features = node.feature.split(",")
            pos = features[0]
            pos_detail1 = features[1]

            tokens.append((surface, pos, pos_detail1))
            surfaces.append(surface)

            if pos == "名詞":
                common_nouns.append(surface)
            if pos in ["名詞", "動詞", "形容詞", "副詞"]:
                common_words.append(surface)

            if surface in first_person_list:
                first_person.append(surface)
            if surface in second_person_list:
                second_person.append(surface)

            if pos == "感動詞" or surface in exclamation_candidates:
                exclamations.append(surface)

            node = node.next

        if len(tokens) >= 3:
            last_phrase = "".join([tok[0] for tok in tokens[-3:]])
            ending_phrases.append(last_phrase)

        if len(tokens) >= 2:
            start_phrase = "".join([tok[0] for tok in tokens[:2]])
            start_phrases.append(start_phrase)

        if len(surfaces) >= 3:
            for i in range(len(surfaces) - 2):
                phrase = "".join(surfaces[i : i + 3])
                speech_patterns.append(phrase)

        for surface, pos, pos_detail1 in tokens:
            if pos == "助詞" and pos_detail1 in ("終助詞", "副助詞／並立助詞／終助詞"):
                final_particles.append(surface)
            elif pos == "助動詞":
                auxiliary_verbs.append(surface)
            elif pos == "動詞" and surface in ("です", "ます"):
                polite_forms.append(surface)

    def filter_counter(counter_list):
        return [(k, v) for k, v in Counter(counter_list).most_common() if v >= min_count][:most_common]

    return {
        "語尾の表現": filter_counter(ending_phrases),
        "話し始めの表現": filter_counter(start_phrases),
        # "定型句（3文字）": filter_counter(speech_patterns),
        "終助詞の使用": filter_counter(final_particles),
        "助動詞の使用": filter_counter(auxiliary_verbs),
        "丁寧語の使用": filter_counter(polite_forms),
        "一人称の傾向": filter_counter(first_person),
        "二人称の傾向": filter_counter(second_person),
        "感嘆詞の傾向": filter_counter(exclamations),
    }


def clean_text(text):
    return text.strip().replace("\u3000", " ").replace("\xa0", " ")


def check_dialogue(dialogue):
    if dialogue:
        return dialogue.strip()
    return None


def parse_html_file(file_path):
    logs = []
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for p in soup.find_all("p"):
        spans = p.find_all("span")
        if len(spans) >= 3:
            char_name = clean_text(spans[1].text)
            dialogue = clean_text(spans[2].text)
        elif len(spans) == 2:
            char_name = clean_text(spans[1].text)
            dialogue = ""
        else:
            continue

        if char_name and dialogue:
            logs.append({"character": char_name, "dialogue": dialogue})
    return logs


def parse_html_plain(file_path):
    logs = []
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    lines = soup.get_text().splitlines()
    pattern = re.compile(r"^\[.*?\]<[^>]+><b>([^<]+)</b>：(.+)$")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            char_name = clean_text(match.group(1))
            dialogue = clean_text(match.group(2))
            logs.append({"character": char_name, "dialogue": dialogue})
    return logs


def parse_txt_file(file_path):
    logs = []
    pattern = re.compile(r"^(.+?)：(.+)$")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                char_name = clean_text(match.group(1))
                dialogue = clean_text(match.group(2))
                logs.append({"character": char_name, "dialogue": dialogue})
    return logs


def character_logs_from_files(file_paths):
    character_logs = {}
    all_logs = []

    for path in tqdm(file_paths):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".html":
            try:
                logs = parse_html_file(path)
                if not logs:  # fallback for plain format HTML
                    logs = parse_html_plain(path)
            except Exception:
                logs = parse_html_plain(path)
        elif ext == ".txt":
            logs = parse_txt_file(path)
        else:
            continue

        for entry in logs:
            char = clean_text(entry["character"])
            dialogue = check_dialogue(entry["dialogue"])
            if dialogue is None:
                continue

            if is_dice_command(dialogue):
                continue

            if char not in character_logs:
                character_logs[char] = []
            character_logs[char].append(dialogue)
            all_logs.append(entry)

    return character_logs, all_logs
