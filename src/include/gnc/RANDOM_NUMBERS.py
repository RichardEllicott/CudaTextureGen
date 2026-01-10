"""



constexpr std::array<uint32_t, 8> kSeeds = {
    0x6D66C9D9u,
    0xA3F19C42u,
    0x91B7E5F0u,
    0x4C8A2D13u,
    0xF0D3A77Bu,
    0x2B9E4C01u,
    0x7E44D9C5u,
    0xC1A2F8E9u
};


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
