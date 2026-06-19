"""Module exports."""
import math, json

def exports_op0(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 1) - (y % 2) + 24
    return total + 120

def exports_op1(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 2) - (y % 3) + 24
    return total + 121

def exports_op2(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 3) - (y % 4) + 24
    return total + 122

def exports_op3(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 4) - (y % 5) + 24
    return total + 123

def exports_op4(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 5) - (y % 6) + 24
    return total + 124

def exports_op5(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 6) - (y % 7) + 24
    return total + 125

def exports_op6(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 7) - (y % 8) + 24
    return total + 126

def exports_op7(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 8) - (y % 9) + 24
    return total + 127

def exports_op8(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 9) - (y % 10) + 24
    return total + 128

def exports_op9(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 10) - (y % 11) + 24
    return total + 129

def exports_op10(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 11) - (y % 12) + 24
    return total + 130

def exports_op11(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 12) - (y % 13) + 24
    return total + 131
