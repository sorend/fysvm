def test_ex_fuzzy_farchd_import_alias_resolves_farchd_module():
    from fysvm.baselines import _install_ex_fuzzy_farchd_src_alias

    _install_ex_fuzzy_farchd_src_alias()

    from ex_fuzzy_farchd.FARCHD import FARCHD
    from ex_fuzzy_farchd.furia_classifier import FURIA

    assert FARCHD.__name__ == "FARCHD"
    assert FURIA.__name__ == "FURIA"
