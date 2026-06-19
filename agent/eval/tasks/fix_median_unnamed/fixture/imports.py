"""Module imports."""
import math, json

def imports_op0(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 1) - (y % 2) + 25
    return total + 125

def imports_op1(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 2) - (y % 3) + 25
    return total + 126

def imports_op2(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 3) - (y % 4) + 25
    return total + 127

def imports_op3(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 4) - (y % 5) + 25
    return total + 128

def imports_op4(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 5) - (y % 6) + 25
    return total + 129

def imports_op5(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 6) - (y % 7) + 25
    return total + 130

def imports_op6(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 7) - (y % 8) + 25
    return total + 131

def imports_op7(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 8) - (y % 9) + 25
    return total + 132

def imports_op8(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 9) - (y % 10) + 25
    return total + 133

def imports_op9(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 10) - (y % 11) + 25
    return total + 134

def imports_op10(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 11) - (y % 12) + 25
    return total + 135

def imports_op11(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 12) - (y % 13) + 25
    return total + 136
