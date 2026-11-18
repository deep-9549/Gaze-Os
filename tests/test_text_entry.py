from __future__ import annotations

import itertools
import string

import pytest

from gazeos.text_entry.decoder import NoisyChannelDecoder
from gazeos.text_entry.layouts import (
    LAYOUTS,
    SYMBOL_PAGES,
    Layout,
    get_layout,
    key_unit_px,
    symbols_layout,
)
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
        assert {k.id for k in lay.keys} >= {"space", "backspace", "speak", "phrases", "enter", "close", "aim"}
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


def test_default_predictor_loads_a_trained_trigram_when_present(tmp_path) -> None:
    from gazeos.text_entry.ngram import KneserNeyTrigram, sentences
    from gazeos.text_entry.predictor import Predictor

    KneserNeyTrigram().fit(sentences("see you tomorrow morning. see you tomorrow night.") * 5).save(tmp_path / "ng.json")
    trained = Predictor.default(get_layout("qwerty"), ngram_path=tmp_path / "ng.json")
    assert "tomorrow" in trained.suggest("see you ", "").words
    seed = Predictor.default(get_layout("qwerty"), ngram_path=tmp_path / "missing.json")
    assert seed.ngram is not None  # falls back to the seed corpus


def _every_layout() -> list[Layout]:
    pages = [symbols_layout(get_layout(name), page) for name in ("qwerty", "alphabetical")
             for page in range(len(SYMBOL_PAGES))]
    return [get_layout(name) for name in LAYOUTS] + pages


@pytest.mark.parametrize("lay", _every_layout(), ids=lambda lay: lay.name)
def test_every_key_is_usable(lay: Layout) -> None:
    """Every key is at least 0.9 of a standard key each way, lies on the keyboard and overlaps no
    other key (the alphabetical layout's Space was 0 wide, so it could not be seen or looked at)."""
    eps = 1e-9
    assert len({k.id for k in lay.keys}) == len(lay.keys)
    for k in lay.keys:
        assert k.w >= 0.9 - eps and k.h >= 0.9 - eps, k
        assert k.x >= -eps and k.y >= -eps and k.x + k.w <= lay.width + eps and k.y + k.h <= lay.height + eps, k
    for a, b in itertools.combinations(lay.keys, 2):
        apart_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x) <= eps
        apart_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y) <= eps
        assert apart_x or apart_y, (a.id, b.id)


@pytest.mark.parametrize("name", ["qwerty", "alphabetical"])
def test_symbol_pages_keep_the_keyboard_in_place(name: str) -> None:
    """The pages of digits and symbols are the letter keyboard's size, so keys keep their size;
    Space, Del, Enter and the rest stay where they were, "ABC" takes 123's place and the page
    switch Shift's. Between them the pages hold every printable ASCII character."""
    base = get_layout(name)
    where = {k.id: (k.x, k.y, k.w, k.h) for k in base.keys}
    chars: set[str] = set()
    for page, switch in enumerate((("symbols", "#+="), ("numbers", "123"))):
        lay = symbols_layout(base, page)
        keys = {k.id: k for k in lay.keys}
        assert (lay.width, lay.height) == (base.width, base.height)
        for kid in ("space", "backspace", "enter", "speak", "phrases", "close", "aim"):
            assert (keys[kid].x, keys[kid].y, keys[kid].w, keys[kid].h) == where[kid], kid
        abc, other = keys["letters"], keys[switch[0]]
        assert abc.label == "ABC" and (abc.x, abc.y, abc.w, abc.h) == where["numbers"]
        assert other.label == switch[1] and (other.x, other.y, other.w, other.h) == where["shift"]
        assert not any(k.letters for k in lay.keys)
        top = sorted((k for k in lay.keys if k.text and k.y == 0.0), key=lambda k: k.x)
        assert "".join(k.text for k in top) == "1234567890"
        chars |= {k.text for k in lay.keys if k.text}
    assert set(string.digits + string.punctuation) <= chars


def test_symbols_are_typed_literally(predictor: Predictor) -> None:
    """Digits and symbols go out exactly once, as labelled: never shifted, no space added after
    '.' or ','. They end the word, use up a pending Shift and leave predictions working."""
    typed: list[str] = []
    lay = get_layout("qwerty")
    sess = TypingSession(lay, predictor, emit=typed.append)
    sym = {k.label: k for k in symbols_layout(lay).keys}
    for ch in "hi":
        sess.press(lay.key_for_letter(ch))  # type: ignore[arg-type]
    sess.press(next(k for k in lay.keys if k.id == "shift"))
    for label in "7@/.,":
        sess.press(sym[label])
    assert sess.text == "hi7@/.," and "".join(typed) == sess.text
    assert not sess.shift and sess.current_word == "" and sess.suggestions.words
    sess.press(lay.key_for_letter("a"))  # type: ignore[arg-type]
    assert sess.text == "hi7@/.,a" and sess.current_word == "a"  # Shift did not wait past the symbols
    sess.press(sym["Space"])
    sess.press(sym["Del"])
    sess.press(sym["Enter"])
    assert sess.text == "hi7@/.,a\n"
