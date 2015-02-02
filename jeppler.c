#include <stdint.h>

#define NN (2) // number of nibbles at a time - 2 is best on my system
               // Each level takes 4x the memory, 2 is 2kB (fits in L1 cache?)
#define N (4*NN)
#define M ((64+N-1)/N)
#define FUDGE (16-(M*NN)%16)
static uint64_t table[1<<N] __attribute__((aligned(64)));
static uint64_t table2[256] __attribute__((aligned(64)));

// saves hand-coding the tables
__attribute__((constructor))
static
void init_table() {
    for(int i0=0; i0<(1<<N); i0++) {
        int i = i0;
        uint64_t k = 0;
        for(int j=0; j<NN; j++) {
            int l = i & 0xf;
            i >>= 4;
            int sh = (15-l) * 4;
            k = k + (UINT64_C(1)<<sh);
        }
        table[i0] = k;
    }

    for(int i=0; i<16; i++) {
        uint64_t k = 0;
        for(int j=0; j<16; j++) {
            table2[i*16+j] = k;
            k = (k << 4) | i;
        }
    }
}

static
uint64_t jepler_nibble_sort_word(uint64_t word) {
    if((word << 4 | word >> 60) == word) return word;
    uint64_t counts = ((uint64_t)(FUDGE)) << 60;
    for(int i=0; i<M; i++) {
        counts += table[(word >> i*N) & ((1<<N)-1)];
    }
    uint64_t w = 0;
    for(int i=15; i>=0; i--) {
        int l = (counts >> (4 * (15-i))) & 0xf;
        w = (w << (4*l)) | table2[i*16+l];
    }
    return w;
}

void nibble_sort_jepler(uint64_t *buf)
{
  for (int i = 0; i < 1024; i++)
    buf[i] = jepler_nibble_sort_word(buf[i]);
}
