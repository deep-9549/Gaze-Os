from __future__ import annotations

import pytest

from gazeos.text_entry.decoder import NoisyChannelDecoder
from gazeos.text_entry.layouts import get_layout, key_unit_px
from gazeos.text_entry.ngram import KneserNeyTrigram, sentences, tokenize
from gazeos.text_entry.phrases import PhraseBank
from gazeos.text_entry.predictor import Predictor
from gazeos.text_entry.session import TypingSession, keystroke_savings_rate
from gazeos.text_entry.trie import PrefixTrie


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor.default(get_layout("qwerty"))


def test_trie_completion() -> None:
    t = PrefixTrie()
    for w, c in (("hello", 5), ("help", 9), ("helmet", 1), ("world", 3)):
        t.add(w, c)
    assert [w for w, _ in t.complete("hel", 2)] == ["help", "hello"]
    assert "world" in t and t.count("nope") == 0
    assert len(t.complete("", 10)) == 4 and len(t) == 4


def test_kneser_ney_is_a_distribution() -> None:
    text = "i need water. i need help. i want water. please call my son. please call the nurse."
    m = KneserNeyTrigram().fit(sentences(text))
    ctx = ["i", "need"]
    total = sum(m.prob(w, ctx) for w in m.vocab)
    assert abs(total - 1.0) < 0.02
    assert m.next_words(["please", "call"], 2)[0][0] in {"my", "the"}
    assert m.prob("water", ["i", "need"]) > m.prob("nurse", ["i", "need"])


def test_ngram_save_load(tmp_path) -> None:
    m = KneserNeyTrigram().fit(sentences("good morning. good night. good morning to you."))
    m.save(tmp_path / "lm.json")
    m2 = KneserNeyTrigram.load(tmp_path / "lm.json")
    assert abs(m.prob("morning", ["good"]) - m2.prob("morning", ["good"])) < 1e-12


def test_tokenize() -> None:
    assert tokenize("I'm FINE, thanks!") == ["i'm", "fine", "thanks", "!"]


def test_layouts() -> None:
    for name in ("qwerty", "alphabetical", "large6"):
        lay = get_layout(name)
        assert all(lay.key_for_letter(ch) is not None for ch in "abcdefghijklmnopqrstuvwxyz")
        assert {k.id for k in lay.keys} >= {"space", "backspace", "speak", "phrases", "enter", "close"}
    assert get_layout("large6").grouped and not get_layout("qwerty").grouped
    assert key_unit_px(40) == 100.0


def test_noisy_channel_fixes_near_miss(predictor: Predictor) -> None:
    """Typing 'hekko' with the gaze landing near 'hello' still decodes to hello."""
    lay = get_layout("qwerty")
    centers = lay.letter_centers()
    # h e k k o, but the two k's are looked at between k and l.
    pts = [centers["h"], centers["e"], (centers["k"][0] + 0.55, centers["k"][1]),
           (centers["k"][0] + 0.55, centers["k"][1]), centers["o"]]
    dec = NoisyChannelDecoder(lay, predictor.trie, sigma_keys=0.5)
    words = [c.word for c in dec.decode(pts, [], predictor.lm_prob, k=3)]
    assert words[0] == "hello"


def test_grouped_layout_is_disambiguated(predictor: Predictor) -> None:
    lay = get_layout("large6")
    dec = NoisyChannelDecoder(lay, predictor.trie, sigma_keys=0.6)
    pts = [lay.key_for_letter(ch).center for ch in "water"]  # type: ignore[union-attr]
    words = [c.word for c in dec.decode(pts, ["i", "need"], predictor.lm_prob, k=5)]
    assert "water" in words


def test_predictions_after_context(predictor: Predictor) -> None:
    s = predictor.suggest("please call my ", "")
    assert len(s.words) == predictor.k
    s2 = predictor.suggest("", "wat")
    assert "water" in s2.words


def test_session_typing_and_choice(predictor: Predictor) -> None:
    typed, keys = [], []
    lay = get_layout("qwerty")
    sess = TypingSession(lay, predictor, emit=typed.append, emit_key=keys.append)
    for ch in "wat":
        sess.press(lay.key_for_letter(ch))  # type: ignore[arg-type]
    assert sess.current_word == "wat"
    sess.choose("water")
    assert sess.text == "water " and keys.count("backspace") == 3
    space = next(k for k in lay.keys if k.id == "space")
    sess.press(space)
    sess.backspace()
    assert sess.text == "water "
    assert keystroke_savings_rate(sess.keystrokes, len("water")) > 0 or sess.keystrokes >= 4


def test_ksr_with_prediction_beats_letters(predictor: Predictor) -> None:
    """KSR on a short AAC-style sentence (PRD 14.6 target: 40% or more)."""
    lay = get_layout("qwerty")
    sess = TypingSession(lay, predictor)
    target = "please call my family "
    for word in target.split():
        for ch in word:
            sugg = sess.suggestions.words if sess.suggestions.words else sess.refresh().words
            if word in sugg:
                break
            sess.press(lay.key_for_letter(ch))  # type: ignore[arg-type]
        sess.choose(word)
    assert sess.text == target
    assert keystroke_savings_rate(sess.keystrokes, len(target)) >= 0.4


def test_learning_boosts_new_words(predictor: Predictor) -> None:
    predictor.learn("gazeos gazeos")
    assert "gazeos" in predictor.trie
    assert "gazeos" in predictor.suggest("", "gaz").words


def test_phrase_bank(tmp_path) -> None:
    pb = PhraseBank(tmp_path / "p.json")
    pb.add("Turn on the fan.")
    pb.used("Turn on the fan.")
    assert pb.ranked()[0] == "Turn on the fan."
    assert not pb.note_typed("see you soon")
    pb.note_typed("see you soon")
    assert pb.note_typed("see you soon")
    assert "see you soon" in PhraseBank(tmp_path / "p.json").phrases
