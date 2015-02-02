#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include "nibble_sort.h"

static long timediff(struct timespec start, struct timespec end) {
  struct timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000L + end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp.tv_sec * 1000000000L + temp.tv_nsec;
}

static const int REPS = 100;
static const int BUFSIZE = 1024;

typedef void (*nibble_func_t)(unsigned long *buf);
typedef struct {
  nibble_func_t func;
  const char *name;
} nibble_sort_t;
static nibble_sort_t funcs[] = {
  {nibble_sort, "ref"},
{simons_nibble_sort_loop, "simons_nibble_loop"},
  {simons_nibble_sort_unrolled, "simons_nibble_sort_unrolled"},
  {simons_nibble_sort_unrolled_8, "simons_nibble_sort_unrolled_8"},
{nibble_sort_jepler, "nibble_sort_jepler"},
};
#define NFUNCS (sizeof(funcs) / sizeof(nibble_sort_t))

int errors[NFUNCS];

static int rand_int(int n) {
  int limit = RAND_MAX - RAND_MAX % n;
  int rnd;
  do {
    rnd = rand();
  } while (rnd >= limit);
  return rnd % n;
}

// from:
// http://stackoverflow.com/questions/3343797/is-this-c-implementation-of-fisher-yates-shuffle-correct
static void shuffle(nibble_sort_t *array, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand_int(i + 1);
    nibble_sort_t tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
}

static unsigned long *getbuf(void) {
  unsigned long *p;
  int res = posix_memalign((void **)&p, 4096, BUFSIZE * 8);
  assert(res == 0);
  assert(p);
  assert(((intptr_t)p & 0xfff) == 0);
  return p;
}

static void validate(unsigned long *test_data) {
  unsigned long *buf = getbuf();
  unsigned long *buf2 = getbuf();
  memcpy(buf2, test_data, BUFSIZE * 8);
  nibble_sort(buf2);
  for (int func = 0; func < NFUNCS; ++func) {
    memcpy(buf, test_data, BUFSIZE * 8);
    funcs[func].func(buf);
    for (int i = 0; i < BUFSIZE; ++i) {
      if (buf[i] != buf2[i]) {
        #ifdef DEBUG
          printf("%s: expected %016lx at %d to sort to %016lx but got %016lx\n",
                 funcs[func].name, test_data[i], i, buf2[i], buf[i]);
        #endif
        errors[func]++;
      }
    }
  }
  free(buf);
  free(buf2);
}

static void timing(unsigned long *test_data) {
  unsigned long *buf = getbuf();
  printf("%16s %16s %16s\n", "ns", "entry name", "errors");
  for (int func = 0; func < NFUNCS; ++func) {
    long times[REPS];
    long best = LONG_MAX;
    for (int i = 0; i < REPS; ++i) {
      memcpy(buf, test_data, BUFSIZE * 8);
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      funcs[func].func(buf);
      clock_gettime(CLOCK_MONOTONIC, &end);
      long t = timediff(start, end);
      times[i] = t;
      if (t < best)
        best = t;
    }
    printf("%16ld %16s %16d\n", best, funcs[func].name, errors[func]);
    #ifdef REPORT_TIMES
      for (int i = 0; i < REPS; ++i)
        printf("%ld ", times[i]);
      printf("\n");
    #endif
  }
  free(buf);
}

int main(void) {
  assert(sizeof(unsigned long) == 8);
  srand(time(NULL) + getpid());

  //shuffle(funcs, NFUNCS);

  unsigned long *pathological_data = getbuf();
  for (int i = 0; i < BUFSIZE; ++i)
    for (int j = 0; j < 8; ++j)
      ((unsigned char *)&pathological_data[i])[j] = i % 1024;
  validate(pathological_data);
  free(pathological_data);

  unsigned long *random_data = getbuf();
  for (int i = 0; i < BUFSIZE * 8; ++i)
    ((unsigned char *)random_data)[i] = rand() & 0xff;
  validate(random_data);
  timing(random_data);
  free(random_data);

  return 0;
}
