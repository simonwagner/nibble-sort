OPT?=-O3
ARCH_FLAGS?=-march=native -mtune=native
CFLAGS+=$(OPT) $(ARCH_FLAGS) -std=gnu99
PERF?=perf

.PHONY=run perf clean

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

nibble_sort: nibble_sort.o jeppler.c run.o ref.o
	$(CC) -o $@ $^ $(CFLAGS)

run: nibble_sort
	./nibble_sort

#note: you must set sysctl kernel.perf_event_paranoid to 0
#and /proc/sys/kernel/kptr_restrict to 0 to use this without
#being root
perf: nibble_sort
	$(PERF) record -o nibble_sort.perf ./nibble_sort
	$(PERF) annotate -i nibble_sort.perf

clean:
	rm -f *.o rm nibble_sort nibble_sort.perf
