"""Module auth — code de remplissage réaliste, sans rapport avec le bug."""
from __future__ import annotations
import math, random, json

def auth_op0(x, y=0):
    """Opération 0 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 1) - (y % (2))
    return total + 28

def auth_op1(x, y=0):
    """Opération 1 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 2) - (y % (3))
    return total + 29

def auth_op2(x, y=0):
    """Opération 2 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 3) - (y % (4))
    return total + 30

def auth_op3(x, y=0):
    """Opération 3 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 4) - (y % (5))
    return total + 31

def auth_op4(x, y=0):
    """Opération 4 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 5) - (y % (6))
    return total + 32

def auth_op5(x, y=0):
    """Opération 5 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 6) - (y % (7))
    return total + 33

def auth_op6(x, y=0):
    """Opération 6 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 7) - (y % (8))
    return total + 34

def auth_op7(x, y=0):
    """Opération 7 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 8) - (y % (9))
    return total + 35

def auth_op8(x, y=0):
    """Opération 8 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 9) - (y % (10))
    return total + 36

def auth_op9(x, y=0):
    """Opération 9 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 10) - (y % (11))
    return total + 37

def auth_op10(x, y=0):
    """Opération 10 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 11) - (y % (12))
    return total + 38

def auth_op11(x, y=0):
    """Opération 11 du module auth."""
    total = 0
    for i in range(abs(int(x)) % 50 + 1):
        total += (i * 12) - (y % (13))
    return total + 39
