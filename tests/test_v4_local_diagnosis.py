from types import SimpleNamespace

from diagnose_alignment_v4_local import resolve_geometry, _scenario_specs


def test_local_diagnosis_geometry_isolation():
    g_t = 4.0
    g_r = 2.0
    local = 1.0

    assert resolve_geometry("registered", local, g_t, g_r) == (0.0, 0.0, 0.0)
    assert resolve_geometry("global_only", local, g_t, g_r) == (4.0, 2.0, 0.0)
    assert resolve_geometry("local_only", local, g_t, g_r) == (0.0, 0.0, 1.0)
    assert resolve_geometry("global_local", local, g_t, g_r) == (4.0, 2.0, 1.0)


def test_default_style_specs_include_one_global_baseline_and_paired_local_levels():
    diagnostic = SimpleNamespace(
        local_scenarios=["registered", "global_only", "local_only", "global_local"],
        local_max_displacements=[0.5, 1.0, 2.0],
    )
    specs = _scenario_specs(diagnostic)
    assert specs == [
        ("registered", 0.0),
        ("global_only", 0.0),
        ("local_only", 0.5),
        ("global_local", 0.5),
        ("local_only", 1.0),
        ("global_local", 1.0),
        ("local_only", 2.0),
        ("global_local", 2.0),
    ]
