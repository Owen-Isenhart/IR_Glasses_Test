from src.identity.state_machine import IdentityStateMachine, StateConfig


def test_evasion_after_sustained_no_face():
    sm = IdentityStateMachine(StateConfig(evasion_frames=3))
    assert sm.update(False, "blocked") == "blocked"
    assert sm.update(False, "blocked") == "blocked"
    assert sm.update(False, "blocked") == "evasion"


def test_match_and_blocked_transitions():
    sm = IdentityStateMachine(StateConfig(evasion_frames=2))
    assert sm.update(True, "matched") == "matched"
    assert sm.update(True, "blocked") == "blocked"
