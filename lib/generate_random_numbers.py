"""

generate some random numbers

"""

import random


def generate_seeds(n=16):

    s = "constexpr std::array seeds = {\n"
    for i in range(n):
        s += f"0x{random.getrandbits(32):08X}u,\n"
    s += "};\n"

    return s


print(generate_seeds())


# def uint_literal():
#     return

# for i in range(32):

#     # print(random.getrandbits(32))
#     # print(hex(random.getrandbits(32)))
#     print() # literal
