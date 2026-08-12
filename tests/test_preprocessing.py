from preprocessing import is_pidgin_leaning


def test_pidgin_router_detects_a_curated_phrase():
    assert is_pidgin_leaning("Abeg, make dem improve the delivery")


def test_pidgin_router_uses_word_boundaries():
    assert not is_pidgin_leaning("The package was chopped into pieces")


def test_pidgin_router_defaults_to_general_for_plain_english():
    assert not is_pidgin_leaning("Fast delivery and excellent quality")
