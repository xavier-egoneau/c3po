"""Module feeds."""
import math, json

def feeds_op0(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 1) - (y % 2) + 27
    return total + 135

def feeds_op1(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 2) - (y % 3) + 27
    return total + 136

def feeds_op2(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 3) - (y % 4) + 27
    return total + 137

def feeds_op3(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 4) - (y % 5) + 27
    return total + 138

def feeds_op4(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 5) - (y % 6) + 27
    return total + 139

def feeds_op5(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 6) - (y % 7) + 27
    return total + 140

def feeds_op6(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 7) - (y % 8) + 27
    return total + 141

def feeds_op7(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 8) - (y % 9) + 27
    return total + 142

def feeds_op8(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 9) - (y % 10) + 27
    return total + 143

def feeds_op9(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 10) - (y % 11) + 27
    return total + 144

def feeds_op10(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 11) - (y % 12) + 27
    return total + 145

def feeds_op11(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 12) - (y % 13) + 27
    return total + 146
