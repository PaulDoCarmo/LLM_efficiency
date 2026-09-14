"""Tests du chargement et du parsing d'ARC-Challenge (voir benchmark.py).

Ne touche ni au GPU ni à un modèle : uniquement le dataset et le parsing.
Lançable tel quel (`.venv/bin/python tests/test_arc_challenge.py`) ou via
pytest si tu l'installes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import (  # noqa: E402
    ARC_CHAT_SYSTEM_PROMPT,
    ARC_SPLIT,
    ARC_TEST_SIZE,
    _arc_chat_context,
    _arc_context,
    arc_gold_index,
    load_arc_challenge,
    prepare_arc_requests,
)

# Chargé une seule fois : le split test entier sert à plusieurs tests.
DOCS = load_arc_challenge()


def test_loads_test_split_only():
    """Le split chargé est bien `test`, et lui seul (1119 items en train et
    299 en validation ne doivent jamais s'y mélanger)."""
    assert str(DOCS.split) == ARC_SPLIT == "test"
    assert len(DOCS) == ARC_TEST_SIZE == 1172


def test_limit_subsamples_without_changing_split():
    assert len(load_arc_challenge(limit=10)) == 10
    # Une limite plus grande que le split ne le fait pas déborder.
    assert len(load_arc_challenge(limit=ARC_TEST_SIZE + 500)) == ARC_TEST_SIZE


def test_answer_key_letter_format():
    doc = {
        "id": "lettre",
        "question": "Q",
        "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
        "answerKey": "C",
    }
    assert arc_gold_index(doc) == 2


def test_answer_key_digit_format():
    """Même question, convention numérique : answerKey "3" doit donner le même
    index que "C" ci-dessus, résolu via choices.label et non via la lettre."""
    doc = {
        "id": "chiffre",
        "question": "Q",
        "choices": {"text": ["a", "b", "c", "d"], "label": ["1", "2", "3", "4"]},
        "answerKey": "3",
    }
    assert arc_gold_index(doc) == 2


def test_answer_key_handles_three_and_five_options():
    three = {
        "id": "trois",
        "question": "Q",
        "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]},
        "answerKey": "1",
    }
    five = {
        "id": "cinq",
        "question": "Q",
        "choices": {"text": ["a", "b", "c", "d", "e"], "label": ["A", "B", "C", "D", "E"]},
        "answerKey": "E",
    }
    assert arc_gold_index(three) == 0
    assert arc_gold_index(five) == 4


def test_unknown_answer_key_raises():
    doc = {
        "id": "inconnu",
        "question": "Q",
        "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": "Z",
    }
    try:
        arc_gold_index(doc)
    except ValueError:
        return
    raise AssertionError("un answerKey absent de choices.label doit lever ValueError")


def test_both_key_formats_present_in_real_split():
    """Les deux conventions existent réellement dans le split test : si l'une
    disparaissait, les tests synthétiques ci-dessus ne prouveraient plus rien
    sur les données réelles."""
    letters = sum(1 for d in DOCS if str(d["answerKey"]).strip().isalpha())
    digits = sum(1 for d in DOCS if str(d["answerKey"]).strip().isdigit())
    assert letters > 0 and digits > 0
    assert letters + digits == ARC_TEST_SIZE


def test_every_real_doc_resolves_to_a_valid_option():
    """Aucune question du split ne doit faire échouer le parsing, et le nombre
    d'options n'est jamais supposé égal à 4."""
    sizes = set()
    for doc in DOCS:
        n_options = len(doc["choices"]["text"])
        assert n_options == len(doc["choices"]["label"])
        assert 0 <= arc_gold_index(doc) < n_options
        sizes.add(n_options)
    assert sizes == {3, 4, 5}


def test_prompt_format():
    """Prompt 0-shot, sans exemple ni lettre d'option."""
    doc = {"question": "Why is the sky blue?"}
    assert _arc_context(doc) == "Question: Why is the sky blue?\nAnswer:"


def test_chat_context_uses_pinned_system_prompt():
    """Le message système est figé dans le code, jamais celui par défaut du
    tokenizer : Qwen2.5 base et Instruct n'ont pas le même, ce qui rendrait
    deux modèles non comparables."""
    from transformers import AutoTokenizer

    doc = {"question": "Why is the sky blue?"}
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ctx = _arc_chat_context(doc, tok)

    assert ARC_CHAT_SYSTEM_PROMPT in ctx
    assert "created by Alibaba Cloud" not in ctx  # le défaut du tokenizer Instruct
    assert doc["question"] in ctx
    # Se termine par l'en-tête du tour assistant, donc l'option se colle sans
    # délimiteur (contrairement au format complétion).
    assert ctx.endswith("<|im_start|>assistant\n")


def test_chat_context_adds_no_leading_space():
    """En ChatML l'option ne doit PAS être précédée d'une espace : le contexte
    finit déjà par un saut de ligne."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    docs = load_arc_challenge(limit=2)
    plain = prepare_arc_requests(tok, docs)
    chat = prepare_arc_requests(tok, docs, chat_template=True)

    assert len(chat["encoded"]) == len(plain["encoded"])
    assert chat["byte_lens"] == plain["byte_lens"]  # acc_norm inchangé
    # Le prompt ChatML est plus long, donc chaque séquence aussi.
    for (pc, pk), (cc, ck) in zip(plain["encoded"], chat["encoded"]):
        assert len(cc) > len(pc)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    sys.exit(1 if failed else 0)
