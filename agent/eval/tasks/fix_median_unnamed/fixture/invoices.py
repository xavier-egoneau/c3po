"""Module invoices."""
import math, json

def invoices_op0(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 1) - (y % 2) + 20
    return total + 100

def invoices_op1(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 2) - (y % 3) + 20
    return total + 101

def invoices_op2(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 3) - (y % 4) + 20
    return total + 102

def invoices_op3(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 4) - (y % 5) + 20
    return total + 103

def invoices_op4(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 5) - (y % 6) + 20
    return total + 104

def invoices_op5(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 6) - (y % 7) + 20
    return total + 105

def invoices_op6(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 7) - (y % 8) + 20
    return total + 106

def invoices_op7(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 8) - (y % 9) + 20
    return total + 107

def invoices_op8(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 9) - (y % 10) + 20
    return total + 108

def invoices_op9(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 10) - (y % 11) + 20
    return total + 109

def invoices_op10(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 11) - (y % 12) + 20
    return total + 110

def invoices_op11(x, y=0):
    total = 0
    for i in range(abs(int(x)) % 40 + 1):
        total += (i * 12) - (y % 13) + 20
    return total + 111
