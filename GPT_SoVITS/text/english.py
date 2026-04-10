import pickle
import importlib.util
import os
import re
import numpy as np

from text.symbols import punctuation

from text.symbols2 import symbols
from text.phone_units import finalize_phone_units, flatten_phone_units

from builtins import str as unicode
from text.en_normalization.expend import normalize

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
NAMECACHE_PATH = os.path.join(current_file_path, "namedict_cache.pickle")
G2P_EN_SPEC = importlib.util.find_spec("g2p_en")
if G2P_EN_SPEC is None or not G2P_EN_SPEC.submodule_search_locations:
    raise ImportError("g2p_en package data files not found")
G2P_EN_DIR = G2P_EN_SPEC.submodule_search_locations[0]
G2P_CHECKPOINT_PATH = os.path.join(G2P_EN_DIR, "checkpoint20.npz")
G2P_HOMOGRAPHS_PATH = os.path.join(G2P_EN_DIR, "homographs.en")
_wordsegment = None
_pos_tag = None


# 适配中文及 g2p_en 标点
rep_map = {
    "[;:：，；]": ",",
    '["’]': "'",
    "。": ".",
    "！": "!",
    "？": "?",
}


arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def construct_homograph_dictionary():
    homograph2features = {}
    with open(G2P_HOMOGRAPHS_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            headword, pron1, pron2, pos1 = line.split("|")
            homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features


def ensure_wordsegment():
    global _wordsegment
    if _wordsegment is None:
        import wordsegment as wordsegment_module

        wordsegment_module.load()
        _wordsegment = wordsegment_module
    return _wordsegment


def ensure_pos_tag():
    global _pos_tag
    if _pos_tag is None:
        from nltk import pos_tag as nltk_pos_tag

        _pos_tag = nltk_pos_tag
    return _pos_tag


def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,?!\-]", text)


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def normalize_pronunciation(pron):
    phones = [ph if ph != "<unk>" else "UNK" for ph in pron if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]
    return replace_phs(phones)


def replace_consecutive_punctuation(text):
    punctuations = "".join(re.escape(p) for p in punctuation)
    pattern = f"([{punctuations}\\s])([{punctuations}])+"
    result = re.sub(pattern, r"\1", text)
    return result


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 57:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0].lower()
                g2p_dict[word] = [word_split[1].split(" ")]

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def hot_reload_hot(g2p_dict):
    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0].lower()
                # 自定义发音词直接覆盖字典
                g2p_dict[word] = [word_split[1:]]

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    g2p_dict = hot_reload_hot(g2p_dict)

    return g2p_dict


def get_namedict():
    if os.path.exists(NAMECACHE_PATH):
        with open(NAMECACHE_PATH, "rb") as pickle_file:
            name_dict = pickle.load(pickle_file)
    else:
        name_dict = {}

    return name_dict


def text_normalize(text):
    # todo: eng text normalize

    # 效果相同，和 chinese.py 保持一致
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    text = pattern.sub(lambda x: rep_map[x.group()], text)

    text = unicode(text)
    text = normalize(text)

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)
    return text


class en_G2p:
    def __init__(self):
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                                             'EY2', 'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 'OW0', 'OW1',
                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 'UW',
                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        self.predictor_loaded = False

        # 扩展过时字典, 添加姓名字典
        self.cmu = get_dict()
        self.namedict = get_namedict()
        self.homograph2features = construct_homograph_dictionary()

        # 剔除读音错误的几个缩写
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # 修正多音字
        self.homograph2features["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
        self.homograph2features["complex"] = (
            ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
            ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
            "JJ",
        )

    def _needs_pos_tag(self, words):
        for original_word in words:
            word = original_word.lower()
            if re.search("[a-z]", word) is None:
                continue
            if word in self.homograph2features:
                return True
        return False

    def _ensure_predictor_loaded(self):
        if self.predictor_loaded:
            return
        variables = np.load(G2P_CHECKPOINT_PATH)
        self.enc_emb = variables["enc_emb"]
        self.enc_w_ih = variables["enc_w_ih"]
        self.enc_w_hh = variables["enc_w_hh"]
        self.enc_b_ih = variables["enc_b_ih"]
        self.enc_b_hh = variables["enc_b_hh"]
        self.dec_emb = variables["dec_emb"]
        self.dec_w_ih = variables["dec_w_ih"]
        self.dec_w_hh = variables["dec_w_hh"]
        self.dec_b_ih = variables["dec_b_ih"]
        self.dec_b_hh = variables["dec_b_hh"]
        self.fc_w = variables["fc_w"]
        self.fc_b = variables["fc_b"]
        self.predictor_loaded = True

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        return x

    def predict(self, word):
        self._ensure_predictor_loaded()
        enc = self.encode(word)
        enc = self.gru(
            enc,
            len(word) + 1,
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32),
        )
        last_hidden = enc[:, -1, :]

        dec = np.take(self.dec_emb, [2], axis=0)
        h = last_hidden

        preds = []
        for _ in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3:
                break
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        return [self.idx2p.get(idx, "<unk>") for idx in preds]

    def __call__(self, text):
        # tokenization
        words = simple_word_tokenize(text)
        tokens = ensure_pos_tag()(words) if self._needs_pos_tag(words) else [(word, "") for word in words]

        # steps
        prons = []
        for o_word, pos in tokens:
            pron = self.pronounce_token(o_word, pos)
            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def pronounce_token(self, o_word, pos):
        word = o_word.lower()

        if re.search("[a-z]", word) is None:
            return [word]
        if len(word) == 1:
            if o_word == "A":
                return ["EY1"]
            return self.cmu[word][0]
        if word in self.homograph2features:
            pron1, pron2, pos1 = self.homograph2features[word]
            if pos.startswith(pos1):
                return pron1
            if len(pos) < len(pos1) and pos == pos1[: len(pos)]:
                return pron1
            return pron2
        return self.qryword(o_word)

    def phone_units(self, text):
        words = simple_word_tokenize(text)
        tokens = ensure_pos_tag()(words) if self._needs_pos_tag(words) else [(word, "") for word in words]
        units = []
        for o_word, pos in tokens:
            pron = self.pronounce_token(o_word, pos)
            unit_type = "punct" if re.search("[a-z]", o_word.lower()) is None else "word"
            units.append(
                {
                    "unit_type": unit_type,
                    "text": o_word,
                    "norm_text": o_word.lower() if unit_type == "word" else o_word,
                    "phones": normalize_pronunciation(pron),
                }
            )
        return finalize_phone_units(units)

    def qryword(self, o_word):
        word = o_word.lower()

        # 查字典, 单字母除外
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # 单词仅首字母大写时查找姓名字典
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # oov 长度小于等于 3 直接读字母
        if len(word) <= 3:
            phones = []
            for w in word:
                # 单读 A 发音修正, 此处不存在大写的情况
                if w == "a":
                    phones.extend(["EY1"])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # 尝试分离所有格
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH 无声辅音结尾 's 发 ['S']
            if phones[-1] in ["P", "T", "K", "F", "TH", "HH"]:
                phones.extend(["S"])
            # S Z SH ZH CH JH 擦声结尾 's 发 ['IH1', 'Z'] 或 ['AH0', 'Z']
            elif phones[-1] in ["S", "Z", "SH", "ZH", "CH", "JH"]:
                phones.extend(["AH0", "Z"])
            # B D G DH V M N NG L R W Y 有声辅音结尾 's 发 ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IY0 IY1 IY2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 元音结尾 's 发 ['Z']
            else:
                phones.extend(["Z"])
            return phones

        # 尝试进行分词，应对复合词
        comps = ensure_wordsegment().segment(word.lower())

        # 无法分词的送回去预测
        if len(comps) == 1:
            return self.predict(word)

        # 可以分词的递归处理
        return [phone for comp in comps for phone in self.qryword(comp)]


_g2p = en_G2p()


def g2p(text):
    return flatten_phone_units(_g2p.phone_units(text))


def g2p_with_phone_units(text):
    units = _g2p.phone_units(text)
    return flatten_phone_units(units), units


if __name__ == "__main__":
    print(g2p("hello"))
    print(g2p(text_normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
