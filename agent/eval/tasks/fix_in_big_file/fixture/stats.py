"""Boîte à outils statistiques."""
import math

def helper_0(seq):
    """Aide 0."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 1) / (i + 1)
    return acc / (len(seq) or 1)

def helper_1(seq):
    """Aide 1."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 2) / (i + 1)
    return acc / (len(seq) or 1)

def helper_2(seq):
    """Aide 2."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 3) / (i + 1)
    return acc / (len(seq) or 1)

def helper_3(seq):
    """Aide 3."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 4) / (i + 1)
    return acc / (len(seq) or 1)

def helper_4(seq):
    """Aide 4."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 5) / (i + 1)
    return acc / (len(seq) or 1)

def helper_5(seq):
    """Aide 5."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 6) / (i + 1)
    return acc / (len(seq) or 1)

def helper_6(seq):
    """Aide 6."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 7) / (i + 1)
    return acc / (len(seq) or 1)

def helper_7(seq):
    """Aide 7."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 8) / (i + 1)
    return acc / (len(seq) or 1)

def helper_8(seq):
    """Aide 8."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 9) / (i + 1)
    return acc / (len(seq) or 1)

def helper_9(seq):
    """Aide 9."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 10) / (i + 1)
    return acc / (len(seq) or 1)

def helper_10(seq):
    """Aide 10."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 11) / (i + 1)
    return acc / (len(seq) or 1)

def helper_11(seq):
    """Aide 11."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 12) / (i + 1)
    return acc / (len(seq) or 1)

def helper_12(seq):
    """Aide 12."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 13) / (i + 1)
    return acc / (len(seq) or 1)

def helper_13(seq):
    """Aide 13."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 14) / (i + 1)
    return acc / (len(seq) or 1)

def helper_14(seq):
    """Aide 14."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 15) / (i + 1)
    return acc / (len(seq) or 1)

def helper_15(seq):
    """Aide 15."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 16) / (i + 1)
    return acc / (len(seq) or 1)

def helper_16(seq):
    """Aide 16."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 17) / (i + 1)
    return acc / (len(seq) or 1)

def helper_17(seq):
    """Aide 17."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 18) / (i + 1)
    return acc / (len(seq) or 1)

def helper_18(seq):
    """Aide 18."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 19) / (i + 1)
    return acc / (len(seq) or 1)

def helper_19(seq):
    """Aide 19."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 20) / (i + 1)
    return acc / (len(seq) or 1)

def helper_20(seq):
    """Aide 20."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 21) / (i + 1)
    return acc / (len(seq) or 1)

def helper_21(seq):
    """Aide 21."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 22) / (i + 1)
    return acc / (len(seq) or 1)

def helper_22(seq):
    """Aide 22."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 23) / (i + 1)
    return acc / (len(seq) or 1)

def helper_23(seq):
    """Aide 23."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 24) / (i + 1)
    return acc / (len(seq) or 1)

def helper_24(seq):
    """Aide 24."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 25) / (i + 1)
    return acc / (len(seq) or 1)

def helper_25(seq):
    """Aide 25."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 26) / (i + 1)
    return acc / (len(seq) or 1)

def helper_26(seq):
    """Aide 26."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 27) / (i + 1)
    return acc / (len(seq) or 1)

def helper_27(seq):
    """Aide 27."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 28) / (i + 1)
    return acc / (len(seq) or 1)

def helper_28(seq):
    """Aide 28."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 29) / (i + 1)
    return acc / (len(seq) or 1)

def helper_29(seq):
    """Aide 29."""
    acc = 0.0
    for i, v in enumerate(seq):
        acc += (v * 30) / (i + 1)
    return acc / (len(seq) or 1)

def median(values):
    s = sorted(values)
    return s[len(s) // 2]  # FIXME pair

def mean(values):
    return sum(values) / len(values)
