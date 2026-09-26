"""Macro F0.5 exactly as the challenge defines it: per Source 1 entity, then averaged.
A singleton (no true matches) scores 1.0 for an empty prediction, 0.0 otherwise."""


def f05(pred: set, true: set) -> float:
    """F0.5 for one Source 1 entity."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict) -> float:
    """pred/truth: {s1_id: set(ids)}; scored over every id in truth."""
    return sum(f05(pred.get(k, set()), v) for k, v in truth.items()) / len(truth)


if __name__ == "__main__":
    # worked example from the problem statement
    assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.714) < 1e-3
    assert f05(set(), set()) == 1.0 and f05({"x"}, set()) == 0.0
    print("ok")
