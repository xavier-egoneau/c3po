"""Module reviews."""
import math, json

def reviews_op0(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 1) - (y % 2) + 14
    return total + 70

def reviews_op1(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 2) - (y % 3) + 14
    return total + 71

def reviews_op2(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 3) - (y % 4) + 14
    return total + 72

def reviews_op3(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 4) - (y % 5) + 14
    return total + 73

def reviews_op4(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 5) - (y % 6) + 14
    return total + 74

def reviews_op5(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 6) - (y % 7) + 14
    return total + 75

def reviews_op6(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 7) - (y % 8) + 14
    return total + 76

def reviews_op7(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 8) - (y % 9) + 14
    return total + 77

def reviews_op8(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 9) - (y % 10) + 14
    return total + 78

def reviews_op9(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 10) - (y % 11) + 14
    return total + 79

def reviews_op10(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 11) - (y % 12) + 14
    return total + 80

def reviews_op11(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 12) - (y % 13) + 14
    return total + 81
