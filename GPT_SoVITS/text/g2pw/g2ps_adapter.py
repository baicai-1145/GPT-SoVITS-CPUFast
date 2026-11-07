"""g2pS adapter: exposes the G2PW converter interface backed by the
2.1 MB numpy-only g2pS nano model (pip install g2ps).

Drop-in for G2PWTorchConverter in chinese2.py:
  __call__(sentences, partial_results) -> List[List[str]]  # per-char tone3 syllables
  polyphonic_chars_new                                    # set of polyphone chars

Fast paths: sentences without any polyphone char skip the neural engine
entirely; the pypinyin dictionary row is fetched once per sentence and only
engine-decided positions overwrite it.
"""
from pypinyin import Style, lazy_pinyin


class G2pSAdapter:
    def __init__(self, model_dir="nano"):
        from g2ps import G2pS
        self.engine = G2pS(model_dir)
        self.polyphonic_chars_new = set(self.engine.char_cands.keys())
        self._poly = self.polyphonic_chars_new
        self.style = "pinyin"

    def _dict_row(self, sent):
        row = lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True,
                          errors="default")
        return row if len(row) == len(sent) else None

    def __call__(self, sentences, partial_results=None):
        if isinstance(sentences, str):
            sentences = [sentences]
        if partial_results is not None and len(partial_results) != len(sentences):
            raise ValueError("partial_results must match sentences length")
        out = []
        for i, sent in enumerate(sentences):
            preset = partial_results[i] if partial_results is not None else None
            if not any(ch in self._poly for ch in sent):
                # no polyphone char: dictionary decides everything
                row = (list(preset) if preset is not None else None) or self._dict_row(sent)
                out.append(row if row else list(sent))
                continue
            base = self._dict_row(sent) or list(sent)
            _, pairs = self.engine.convert(sent)
            row = list(base)
            for j, (ch, py) in enumerate(pairs):
                if preset is not None and j < len(preset) and preset[j] is not None:
                    row[j] = preset[j]
                elif py is not None and j < len(row):
                    row[j] = py.replace("u:", "v")
            out.append(row)
        return out
