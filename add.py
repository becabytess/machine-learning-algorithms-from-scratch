def factorial(num):

    sum = 1
    if num == 0:
        return 1
    while num > 1:
        sum *= num
        num -= 1
    return sum


def expo(num, power):
    sum = 0
    while power > -1:
        sum += num ** power / factorial(power)
        power -= 1
    return sum


print(expo(4, 20))  