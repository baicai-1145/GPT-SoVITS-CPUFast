import importlib.util
import os
import pickle
import re
import sys
from typing import Any, Dict

import jieba_fast
from jieba_fast._compat import PY2, default_encoding, resolve_filename, strdecode

POSSEG_CACHE_VERSION = 1

re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile("(\r\n|\s)")

re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")
re_eng1 = re.compile("^[a-zA-Z0-9]$", re.U)

current_file_path = os.path.dirname(__file__)
POSSEG_CACHE_PATH = os.path.join(current_file_path, "jieba_posseg_assets_v1.pkl")


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_posseg_tables():
    posseg_dir = os.path.join(os.path.dirname(jieba_fast.__file__), "posseg")
    char_state_tab = _load_module_from_path("_gptsovits_char_state_tab", os.path.join(posseg_dir, "char_state_tab.py"))
    prob_start = _load_module_from_path("_gptsovits_prob_start", os.path.join(posseg_dir, "prob_start.py"))
    prob_trans = _load_module_from_path("_gptsovits_prob_trans", os.path.join(posseg_dir, "prob_trans.py"))
    prob_emit = _load_module_from_path("_gptsovits_prob_emit", os.path.join(posseg_dir, "prob_emit.py"))
    return char_state_tab.P, prob_start.P, prob_trans.P, prob_emit.P


char_state_tab_P, start_P, trans_P, emit_P = _load_posseg_tables()


MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    mem_path = [{}]
    all_states = trans_p.keys()
    for y in states.get(obs[0], all_states):
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ""
    for t in range(1, len(obs)):
        V.append({})
        mem_path.append({})
        prev_states = [x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]

        prev_states_expect_next = set((y for x in prev_states for y in trans_p[x].keys()))
        obs_states = set(states.get(obs[t], all_states)) & prev_states_expect_next

        if not obs_states:
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states

        for y in obs_states:
            prob, state = max(
                (
                    V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) + emit_p[y].get(obs[t], MIN_FLOAT),
                    y0,
                )
                for y0 in prev_states
            )
            V[t][y] = prob
            mem_path[t][y] = state

    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    prob, state = max(last)

    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    return prob, route


def _build_assets(tokenizer) -> Dict[str, Any]:
    tokenizer.initialize()
    dict_file = tokenizer.get_dict_file()
    dict_path = resolve_filename(dict_file)

    if isinstance(dict_file, str):
        f = open(dict_file, "rb")
    else:
        f = tokenizer.get_dict_file()

    word_tag_tab = {}
    for line in f:
        line = line.strip().decode("utf-8")
        if not line:
            continue
        word, _, tag = line.split(" ")
        word_tag_tab[word] = tag
    f.close()

    return {
        "cache_version": POSSEG_CACHE_VERSION,
        "dict_path": dict_path,
        "dict_mtime": os.path.getmtime(dict_path),
        "FREQ": tokenizer.FREQ,
        "total": tokenizer.total,
        "word_tag_tab": word_tag_tab,
    }


def _load_or_build_assets(tokenizer) -> Dict[str, Any]:
    dict_path = resolve_filename(tokenizer.get_dict_file())
    dict_mtime = os.path.getmtime(dict_path)
    if os.path.exists(POSSEG_CACHE_PATH):
        try:
            with open(POSSEG_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if (
                cached.get("cache_version") == POSSEG_CACHE_VERSION
                and cached.get("dict_path") == dict_path
                and cached.get("dict_mtime") == dict_mtime
            ):
                return cached
        except Exception:
            pass

    built = _build_assets(tokenizer)
    try:
        with open(POSSEG_CACHE_PATH, "wb") as f:
            pickle.dump(built, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return built


class pair(object):
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return "%s/%s" % (self.word, self.flag)

    def __repr__(self):
        return "pair(%r, %r)" % (self.word, self.flag)

    def __str__(self):
        if PY2:
            return self.__unicode__().encode(default_encoding)
        return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other, pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)

    def encode(self, arg):
        return self.__unicode__().encode(arg)


class POSTokenizer(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or jieba_fast.Tokenizer()
        self._load_cached_assets()

    def __repr__(self):
        return "<POSTokenizer tokenizer=%r>" % self.tokenizer

    def __getattr__(self, name):
        if name in ("cut_for_search", "lcut_for_search", "tokenize"):
            raise NotImplementedError
        return getattr(self.tokenizer, name)

    def _load_cached_assets(self):
        assets = _load_or_build_assets(self.tokenizer)
        self.tokenizer.FREQ = assets["FREQ"]
        self.tokenizer.total = assets["total"]
        self.tokenizer.initialized = True
        self.word_tag_tab = assets["word_tag_tab"]

    def initialize(self, dictionary=None):
        if dictionary:
            self.tokenizer.initialize(dictionary)
            self.load_word_tag(self.tokenizer.get_dict_file())
            return
        self._load_cached_assets()

    def load_word_tag(self, f):
        self.word_tag_tab = {}
        f_name = resolve_filename(f)
        for lineno, line in enumerate(f, 1):
            try:
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                word, _, tag = line.split(" ")
                self.word_tag_tab[word] = tag
            except Exception:
                raise ValueError("invalid POS dictionary entry in %s at Line %s: %s" % (f_name, lineno, line))
        f.close()

    def makesure_userdict_loaded(self):
        if self.tokenizer.user_word_tag_tab:
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            self.tokenizer.user_word_tag_tab = {}

    def __cut(self, sentence):
        prob, pos_list = viterbi(sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            if pos == "B":
                begin = i
            elif pos == "E":
                yield pair(sentence[begin : i + 1], pos_list[i][1])
                nexti = i + 1
            elif pos == "S":
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    def __cut_detail(self, sentence):
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            if re_han_detail.match(blk):
                for word in self.__cut(blk):
                    yield word
            else:
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        if re_num.match(x):
                            yield pair(x, "m")
                        elif re_eng.match(x):
                            yield pair(x, "eng")
                        else:
                            yield pair(x, "x")

    def __cut_DAG_NO_HMM(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ""
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield pair(buf, "eng")
                    buf = ""
                yield pair(l_word, self.word_tag_tab.get(l_word, "x"))
                x = y
        if buf:
            yield pair(buf, "eng")

    def __cut_DAG(self, sentence):
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}

        self.tokenizer.calc(sentence, DAG, route)

        x = 0
        buf = ""
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield pair(buf, self.word_tag_tab.get(buf, "x"))
                    elif not self.tokenizer.FREQ.get(buf):
                        recognized = self.__cut_detail(buf)
                        for t in recognized:
                            yield t
                    else:
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, "x"))
                    buf = ""
                yield pair(l_word, self.word_tag_tab.get(l_word, "x"))
            x = y

        if buf:
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, "x"))
            elif not self.tokenizer.FREQ.get(buf):
                recognized = self.__cut_detail(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, "x"))

    def __cut_internal(self, sentence, HMM=True):
        self.makesure_userdict_loaded()
        sentence = strdecode(sentence)
        blocks = re_han_internal.split(sentence)
        cut_blk = self.__cut_DAG if HMM else self.__cut_DAG_NO_HMM

        for blk in blocks:
            if re_han_internal.match(blk):
                for word in cut_blk(blk):
                    yield word
            else:
                tmp = re_skip_internal.split(blk)
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield pair(x, "x")
                    else:
                        for xx in x:
                            if re_num.match(xx):
                                yield pair(xx, "m")
                            elif re_eng.match(x):
                                yield pair(xx, "eng")
                            else:
                                yield pair(xx, "x")

    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w

    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))


dt = POSTokenizer(jieba_fast.dt)
initialize = dt.initialize


def _lcut_internal(s):
    return dt._lcut_internal(s)


def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)


def cut(sentence, HMM=True):
    global dt
    if jieba_fast.pool is None:
        for w in dt.cut(sentence, HMM=HMM):
            yield w
    else:
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = jieba_fast.pool.map(_lcut_internal, parts)
        else:
            result = jieba_fast.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w


def lcut(sentence, HMM=True):
    return list(cut(sentence, HMM))
