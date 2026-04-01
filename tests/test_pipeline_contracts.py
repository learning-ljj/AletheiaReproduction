from src.core.config import load_prompts


def test_prompt_generator_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["generator"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "<lemma>" in text
    assert "</lemma>" in text
    assert "[cite:" in text



def test_prompt_verifier_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["verifier"]["phase3_user"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<verification>" in text
    assert "</verification>" in text
    assert "<verified_lemmas>" in text
    assert "</verified_lemmas>" in text
    assert "<citation_review>" in text
    assert "</citation_review>" in text



def test_prompt_reviser_contract_tags() -> None:
    prompts = load_prompts()
    text = prompts["reviser"]["system"]

    assert "<verdict>" in text
    assert "</verdict>" in text
    assert "<solution>" in text
    assert "</solution>" in text
    assert "[cite:" in text
