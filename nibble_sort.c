#include <emmintrin.h>
#include <xmmintrin.h>
#include <tmmintrin.h>

/*
Nibble sort implementation using count sort

The basic idea is the following: Load the 64bit qword into
an XMM register, then XOR it with NIBBLE ^ 1 (which should give 1
if the nibble is NIBBLE), then do popcnt
*/

#define EXPAND_HEXNIBBLE_64(n) 0x ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## n ## UL
#define EXPAND_HEXNIBBLE_128(n) { EXPAND_HEXNIBBLE_64(n), EXPAND_HEXNIBBLE_64(n) }
#define NIBBLE_MASK(n) EXPAND_HEXNIBBLE_64(n)
#define NIBBLE_MASK_128(n) EXPAND_HEXNIBBLE_128(n)
#define NIBBLE_BITSET(n, xor_nibble, reg) \
  __m128i bitset_ ## n; \
  { \
    __m128i xor_mask = NIBBLE_MASK_128(xor_nibble); \
    reg = _mm_xor_si128(reg, xor_mask); \
    bitset_ ## n = zero_bitset(reg); \
  } \

#define NIBBLE_COUNT(n) __m128i count_ ## n  = popcnt_epi64(bitset_ ## n);
#define NIBBLE_COUNT_COMPRESSED(n) count = count | (popcnt_epi64(bitset_ ## n) << ((n)*5))
#define SORTED_NIBBLE_FUNC(output, nibble, index, sorted_nibble) \
  { \
    unsigned long nibble_mask = NIBBLE_MASK(nibble); \
    __m128i count_oword = count_ ## nibble; \
\
    output[0] |= sorted_nibble(nibble_mask, index[0], count_oword[0]); \
    output[1] |= sorted_nibble(nibble_mask, index[1], count_oword[1]); \
    index[0] += count_oword[0]; \
    index[1] += count_oword[1]; \
  } \

//per default, use sorted_nibble
#define SORTED_NIBBLE(output, nibble, index) SORTED_NIBBLE_FUNC(output, nibble, index, sorted_nibble)
//for the last ones (who would result in undefined behaviour because we might have an overflow)
#define SORTED_NIBBLE_END(output, nibble, index) SORTED_NIBBLE_FUNC(output, nibble, index, sorted_nibble_end)

inline static __m128i zero_bitset(const __m128i input) {
  /*static const __m128i ones = EXPAND_HEXNIBBLE_128(1);
  static const __m128i eights = EXPAND_HEXNIBBLE_128(8);
  __m128i highbit_set = _mm_sub_epi64(input, ones);
  __m128i combinedset = _mm_andnot_si128(input, highbit_set);
  __m128i clearedset = _mm_and_si128(combinedset, eights);

  return clearedset;*/

  //johannes' version
  // ((mask - (input & ~mask)) & ~input & mask)
  static const __m128i mask = EXPAND_HEXNIBBLE_128(8);
  __m128i input_and_not_mask = _mm_andnot_si128(mask, input);
  __m128i not_input_and_mask = _mm_andnot_si128(input, mask);
  __m128i mask_minus_input_and_not_mask = _mm_sub_epi64(mask, input_and_not_mask);
  __m128i final = _mm_and_si128(mask_minus_input_and_not_mask, not_input_and_mask);

  return final;
}

inline static __m128i nibble_bitset(const __m128i input, const __m128i nibble_mask) {
  __m128i nibble_zeroed = _mm_xor_si128(input, nibble_mask);
  return zero_bitset(nibble_zeroed);
}

/*
custom popcnt, will do popcnt in paralell for 2 qwords

stolen from: http://wm.ite.pl/articles/sse-popcount.html
*/
inline static __m128i popcnt_epi64(const __m128i input) {
  static const __m128i odd_nibble_mask = {
    0x0f0f0f0f0f0f0f0fUL,
    0x0f0f0f0f0f0f0f0fUL
  };

  /*
  lookup table for popularity count
  maps bytes from 0x00 to 0x0f to their bitcount
     0 -> 0,
  	 1 -> 1,
  	 2 -> 1,
  	 3 -> 2,
  	 4 -> 1,
  	 5 -> 2,
  	 6 -> 2,
  	 7 -> 3,
  	 8 -> 1,
  	 9 -> 2,
  	 a -> 2,
  	 b -> 3,
  	 c -> 2,
  	 d -> 3,
  	 e -> 3,
  	 f -> 4
  */
  static const __m128i popcnt_lut = { 0x0302020102010100UL,
                                      0x0403030203020201UL };

  __m128i shifted_left = _mm_srli_epi64(input, 4);
  __m128i lower = _mm_and_si128(input, odd_nibble_mask);
  __m128i upper = _mm_and_si128(shifted_left, odd_nibble_mask);

  __m128i lower_popcnt_epi4 = _mm_shuffle_epi8(popcnt_lut, lower);
  __m128i upper_popcnt_epi4 = _mm_shuffle_epi8(popcnt_lut, upper);

  __m128i popcnt_epi8 = _mm_add_epi8(lower_popcnt_epi4, upper_popcnt_epi4);

  __m128i zero = {0, 0};
  __m128i popcnt_epi64 = _mm_sad_epu8(popcnt_epi8, zero);

  return popcnt_epi64;
}


inline static unsigned long sorted_nibble(unsigned long nibble_mask, unsigned int pos, unsigned int count) {
  static unsigned long long mask_lut[] = {
    0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,0x0000000000000000ULL,
    0x000000000000000FULL,0x00000000000000F0ULL,0x0000000000000F00ULL,0x000000000000F000ULL,0x00000000000F0000ULL,0x0000000000F00000ULL,0x000000000F000000ULL,0x00000000F0000000ULL,0x0000000F00000000ULL,0x000000F000000000ULL,0x00000F0000000000ULL,0x0000F00000000000ULL,0x000F000000000000ULL,0x00F0000000000000ULL,0x0F00000000000000ULL,0xF000000000000000ULL,
    0x00000000000000FFULL,0x0000000000000FF0ULL,0x000000000000FF00ULL,0x00000000000FF000ULL,0x0000000000FF0000ULL,0x000000000FF00000ULL,0x00000000FF000000ULL,0x0000000FF0000000ULL,0x000000FF00000000ULL,0x00000FF000000000ULL,0x0000FF0000000000ULL,0x000FF00000000000ULL,0x00FF000000000000ULL,0x0FF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x0000000000000FFFULL,0x000000000000FFF0ULL,0x00000000000FFF00ULL,0x0000000000FFF000ULL,0x000000000FFF0000ULL,0x00000000FFF00000ULL,0x0000000FFF000000ULL,0x000000FFF0000000ULL,0x00000FFF00000000ULL,0x0000FFF000000000ULL,0x000FFF0000000000ULL,0x00FFF00000000000ULL,0x0FFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x000000000000FFFFULL,0x00000000000FFFF0ULL,0x0000000000FFFF00ULL,0x000000000FFFF000ULL,0x00000000FFFF0000ULL,0x0000000FFFF00000ULL,0x000000FFFF000000ULL,0x00000FFFF0000000ULL,0x0000FFFF00000000ULL,0x000FFFF000000000ULL,0x00FFFF0000000000ULL,0x0FFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x00000000000FFFFFULL,0x0000000000FFFFF0ULL,0x000000000FFFFF00ULL,0x00000000FFFFF000ULL,0x0000000FFFFF0000ULL,0x000000FFFFF00000ULL,0x00000FFFFF000000ULL,0x0000FFFFF0000000ULL,0x000FFFFF00000000ULL,0x00FFFFF000000000ULL,0x0FFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x0000000000FFFFFFULL,0x000000000FFFFFF0ULL,0x00000000FFFFFF00ULL,0x0000000FFFFFF000ULL,0x000000FFFFFF0000ULL,0x00000FFFFFF00000ULL,0x0000FFFFFF000000ULL,0x000FFFFFF0000000ULL,0x00FFFFFF00000000ULL,0x0FFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x000000000FFFFFFFULL,0x00000000FFFFFFF0ULL,0x0000000FFFFFFF00ULL,0x000000FFFFFFF000ULL,0x00000FFFFFFF0000ULL,0x0000FFFFFFF00000ULL,0x000FFFFFFF000000ULL,0x00FFFFFFF0000000ULL,0x0FFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x00000000FFFFFFFFULL,0x0000000FFFFFFFF0ULL,0x000000FFFFFFFF00ULL,0x00000FFFFFFFF000ULL,0x0000FFFFFFFF0000ULL,0x000FFFFFFFF00000ULL,0x00FFFFFFFF000000ULL,0x0FFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x0000000FFFFFFFFFULL,0x000000FFFFFFFFF0ULL,0x00000FFFFFFFFF00ULL,0x0000FFFFFFFFF000ULL,0x000FFFFFFFFF0000ULL,0x00FFFFFFFFF00000ULL,0x0FFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x000000FFFFFFFFFFULL,0x00000FFFFFFFFFF0ULL,0x0000FFFFFFFFFF00ULL,0x000FFFFFFFFFF000ULL,0x00FFFFFFFFFF0000ULL,0x0FFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x00000FFFFFFFFFFFULL,0x0000FFFFFFFFFFF0ULL,0x000FFFFFFFFFFF00ULL,0x00FFFFFFFFFFF000ULL,0x0FFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x0000FFFFFFFFFFFFULL,0x000FFFFFFFFFFFF0ULL,0x00FFFFFFFFFFFF00ULL,0x0FFFFFFFFFFFF000ULL,0xFFFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x000FFFFFFFFFFFFFULL,0x00FFFFFFFFFFFFF0ULL,0x0FFFFFFFFFFFFF00ULL,0xFFFFFFFFFFFFF000ULL,0xFFFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x00FFFFFFFFFFFFFFULL,0x0FFFFFFFFFFFFFF0ULL,0xFFFFFFFFFFFFFF00ULL,0xFFFFFFFFFFFFF000ULL,0xFFFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0x0FFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFF0ULL,0xFFFFFFFFFFFFFF00ULL,0xFFFFFFFFFFFFF000ULL,0xFFFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,
    0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFF0ULL,0xFFFFFFFFFFFFFF00ULL,0xFFFFFFFFFFFFF000ULL,0xFFFFFFFFFFFF0000ULL,0xFFFFFFFFFFF00000ULL,0xFFFFFFFFFF000000ULL,0xFFFFFFFFF0000000ULL,0xFFFFFFFF00000000ULL,0xFFFFFFF000000000ULL,0xFFFFFF0000000000ULL,0xFFFFF00000000000ULL,0xFFFF000000000000ULL,0xFFF0000000000000ULL,0xFF00000000000000ULL,0xF000000000000000ULL,

  };

  return nibble_mask & mask_lut[count*16 + (pos % 16)];
}

static const unsigned short SIZE = 1024;


extern void simons_nibble_sort_unrolled_8(unsigned long *buf) {

  for(unsigned short i = 0; i < SIZE; i+=2) {
    __m128i qwords = _mm_load_si128((const __m128i*)(&buf[i]));
    unsigned long sorted_qwords[2] = {0, 0};
    char index[2] = {0, 0};


    /*
    Theoretically, I would have to xor qwords with each NIBBLE_MASK(n) in 0 to f
    As xoring destroys the register, I would have to restore the register.
    I do not want to do that.
    Luckily XOR is reversible so when I do x' = x ^ mask, i can restore x by doing x' ^ mask
    So for x'n+1 I will do x'n+1 = x'n ^ NIBBLE_MASK(n) ^ NIBBLE_MASK(n+1).
    NIBBLE_MASK(n) ^ NIBBLE_MASK(n+1) can be precalculated, so we have the same work
    but we can save a register and a copy operation to restore x.

    Precalculated xor masks:
    ['0x0', '0x1', '0x3', '0x1', '0x7', '0x1', '0x3', '0x1', '0xf', '0x1', '0x3', '0x1', '0x7', '0x1', '0x3', '0x1']
    */

    //generate bitset for 0
    __m128i bitset_0 = zero_bitset(qwords);
    //generate bitsets for 1 - 7
    NIBBLE_BITSET(1, 1, qwords);
    NIBBLE_BITSET(2, 3, qwords);
    NIBBLE_BITSET(3, 1, qwords);
    NIBBLE_BITSET(4, 7, qwords);
    NIBBLE_BITSET(5, 1, qwords);
    NIBBLE_BITSET(6, 3, qwords);
    NIBBLE_BITSET(7, 1, qwords);

    //now count the nibbles
    NIBBLE_COUNT(0)
    NIBBLE_COUNT(1)
    NIBBLE_COUNT(2)
    NIBBLE_COUNT(3)
    NIBBLE_COUNT(4)
    NIBBLE_COUNT(5)
    NIBBLE_COUNT(6)
    NIBBLE_COUNT(7)

    //after counting the nibbles, build the first part of the sorted
    //qword
    SORTED_NIBBLE(sorted_qwords, 0, index)
    SORTED_NIBBLE(sorted_qwords, 1, index)
    SORTED_NIBBLE(sorted_qwords, 2, index)
    SORTED_NIBBLE(sorted_qwords, 3, index)
    SORTED_NIBBLE(sorted_qwords, 4, index)
    SORTED_NIBBLE(sorted_qwords, 5, index)
    SORTED_NIBBLE(sorted_qwords, 6, index)
    SORTED_NIBBLE(sorted_qwords, 7, index)


    //now do it all again for 8 to f

    //generate bitsets for 8 - f
    NIBBLE_BITSET(8, f, qwords);
    NIBBLE_BITSET(9, 1, qwords);
    NIBBLE_BITSET(a, 3, qwords);
    NIBBLE_BITSET(b, 1, qwords);
    NIBBLE_BITSET(c, 7, qwords);
    NIBBLE_BITSET(d, 1, qwords);
    NIBBLE_BITSET(e, 3, qwords);
    NIBBLE_BITSET(f, 1, qwords);

    //now count the nibbles
    NIBBLE_COUNT(8)
    NIBBLE_COUNT(9)
    NIBBLE_COUNT(a)
    NIBBLE_COUNT(b)
    NIBBLE_COUNT(c)
    NIBBLE_COUNT(d)
    NIBBLE_COUNT(e)
    NIBBLE_COUNT(f)

    //after counting the nibbles, build the first part of the sorted
    //qword
    SORTED_NIBBLE(sorted_qwords, 8, index)
    SORTED_NIBBLE(sorted_qwords, 9, index)
    SORTED_NIBBLE(sorted_qwords, a, index)
    SORTED_NIBBLE(sorted_qwords, b, index)
    SORTED_NIBBLE(sorted_qwords, c, index)
    SORTED_NIBBLE(sorted_qwords, d, index)
    SORTED_NIBBLE(sorted_qwords, e, index)
    SORTED_NIBBLE(sorted_qwords, f, index)



    //write back
    buf[i] = sorted_qwords[0];
    buf[i+1] = sorted_qwords[1];
  }
}

extern void simons_nibble_sort_unrolled(unsigned long *buf) {

  for(unsigned short i = 0; i < SIZE; i+=2) {
    __m128i qwords = _mm_load_si128((const __m128i*)(&buf[i]));
    unsigned long sorted_qwords[2] = {0, 0};
    char index[2] = {0, 0};


    /*
    Theoretically, I would have to xor qwords with each NIBBLE_MASK(n) in 0 to f
    As xoring destroys the register, I would have to restore the register.
    I do not want to do that.
    Luckily XOR is reversible so when I do x' = x ^ mask, i can restore x by doing x' ^ mask
    So for x'n+1 I will do x'n+1 = x'n ^ NIBBLE_MASK(n) ^ NIBBLE_MASK(n+1).
    NIBBLE_MASK(n) ^ NIBBLE_MASK(n+1) can be precalculated, so we have the same work
    but we can save a register and a copy operation to restore x.

    Precalculated xor masks:
    ['0x0', '0x1', '0x3', '0x1', '0x7', '0x1', '0x3', '0x1', '0xf', '0x1', '0x3', '0x1', '0x7', '0x1', '0x3', '0x1']
    */
    __m128i bitset_0 = zero_bitset(qwords);
    NIBBLE_COUNT(0)
    SORTED_NIBBLE(sorted_qwords, 0, index)
    NIBBLE_BITSET(1, 1, qwords);
    NIBBLE_COUNT(1)
    SORTED_NIBBLE(sorted_qwords, 1, index)
    NIBBLE_BITSET(2, 3, qwords);
    NIBBLE_COUNT(2)
    SORTED_NIBBLE(sorted_qwords, 2, index)
    NIBBLE_BITSET(3, 1, qwords);
    NIBBLE_COUNT(3)
    SORTED_NIBBLE(sorted_qwords, 3, index)
    NIBBLE_BITSET(4, 7, qwords);
    NIBBLE_COUNT(4)
    SORTED_NIBBLE(sorted_qwords, 4, index)
    NIBBLE_BITSET(5, 1, qwords);
    NIBBLE_COUNT(5)
    SORTED_NIBBLE(sorted_qwords, 5, index)
    NIBBLE_BITSET(6, 3, qwords);
    NIBBLE_COUNT(6)
    SORTED_NIBBLE(sorted_qwords, 6, index)
    NIBBLE_BITSET(7, 1, qwords);
    NIBBLE_COUNT(7)
    SORTED_NIBBLE(sorted_qwords, 7, index)
    NIBBLE_BITSET(8, f, qwords);
    NIBBLE_COUNT(8)
    SORTED_NIBBLE(sorted_qwords, 8, index)
    NIBBLE_BITSET(9, 1, qwords);
    NIBBLE_COUNT(9)
    SORTED_NIBBLE(sorted_qwords, 9, index)
    NIBBLE_BITSET(a, 3, qwords);
    NIBBLE_COUNT(a)
    SORTED_NIBBLE(sorted_qwords, a, index)
    NIBBLE_BITSET(b, 1, qwords);
    NIBBLE_COUNT(b)
    SORTED_NIBBLE(sorted_qwords, b, index)
    NIBBLE_BITSET(c, 7, qwords);
    NIBBLE_COUNT(c)
    SORTED_NIBBLE(sorted_qwords, c, index)
    NIBBLE_BITSET(d, 1, qwords);
    NIBBLE_COUNT(d)
    SORTED_NIBBLE(sorted_qwords, d, index)
    NIBBLE_BITSET(e, 3, qwords);
    NIBBLE_COUNT(e)
    SORTED_NIBBLE(sorted_qwords, e, index)
    NIBBLE_BITSET(f, 1, qwords);
    NIBBLE_COUNT(f)
    SORTED_NIBBLE(sorted_qwords, f, index)


    //write back
    buf[i] = sorted_qwords[0];
    buf[i+1] = sorted_qwords[1];
  }
}

extern void simons_nibble_sort_loop(unsigned long *buf) {
  for(unsigned short i = 0; i < SIZE; i+=2) {
    __m128i qwords = _mm_load_si128((const __m128i*)(&buf[i]));
    __m128i sorted_qwords = {0UL, 0UL};
    __m128i index = {0UL, 0UL};

    unsigned long nibble_mask = 0x11111111111111ULL;

    static const unsigned long NIBBLE_MASKS[] = {
      EXPAND_HEXNIBBLE_64(0),
      EXPAND_HEXNIBBLE_64(1),
      EXPAND_HEXNIBBLE_64(2),
      EXPAND_HEXNIBBLE_64(3),
      EXPAND_HEXNIBBLE_64(4),
      EXPAND_HEXNIBBLE_64(5),
      EXPAND_HEXNIBBLE_64(6),
      EXPAND_HEXNIBBLE_64(7),
      EXPAND_HEXNIBBLE_64(8),
      EXPAND_HEXNIBBLE_64(9),
      EXPAND_HEXNIBBLE_64(a),
      EXPAND_HEXNIBBLE_64(b),
      EXPAND_HEXNIBBLE_64(c),
      EXPAND_HEXNIBBLE_64(d),
      EXPAND_HEXNIBBLE_64(e),
      EXPAND_HEXNIBBLE_64(f),
    };

    for(unsigned int nibble=0x0; nibble <= 0xf; nibble += 0x1) {
      __m128i xor_mask = {NIBBLE_MASKS[nibble], NIBBLE_MASKS[nibble]};

      __m128i nibble_zeroed = _mm_xor_si128(qwords, xor_mask);
      __m128i bitset = zero_bitset(nibble_zeroed);

      __m128i count  = popcnt_epi64(bitset);
      unsigned long nibble_mask = NIBBLE_MASKS[nibble];
      //sorted_qwords |= sorted_nibble_si128(xor_mask, index, count);
      __m128i sorted_nibble_mask = {
        sorted_nibble(xor_mask[0], index[0], count[0]),
        sorted_nibble(xor_mask[1], index[1], count[1])
      };
      sorted_qwords |= sorted_nibble_mask;
      index += count;
    }

    buf[i] = sorted_qwords[0];
    buf[i+1] = sorted_qwords[1];
  }
}

#ifdef TEST

/*#include <stdio.h>

int main(int argc, char** argv) {
  __m128i test = {0b01101, 0b1110110};
  __m128i popcnt = popcnt_epi64(test);

  printf("popcnt: %lld/%lld\n", popcnt[0], popcnt[1]);
}*/
#endif
