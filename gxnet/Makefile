
CFLAGS = -std=c++11 -Wall -Werror

ifeq ($(debug),1)
CFLAGS += -g
else
CFLAGS += -O3 -DNDEBUG
endif

CPPFLAGS = $(CFLAGS)

LDFLAGS = -lstdc++ -lm

######################################################################

PROGS = gxocr 

TEST_PROGS = testbackward testcnn testseeds \
	testmnist testemnist

######################################################################

COMM_OBJS = gxeval.o gxutils.o gxact.o gxlayer.o gxnet.o

######################################################################

all: $(PROGS) $(TEST_PROGS)

#=====================================================================

gxocr: $(COMM_OBJS) gxocr.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

testbackward: $(COMM_OBJS) testbackward.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

testcnn: $(COMM_OBJS) testcnn.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

testseeds: $(COMM_OBJS) testseeds.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

testmnist: $(COMM_OBJS) testmnist.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

testemnist: $(COMM_OBJS) testemnist.o
	gcc $(CFLAGS) -o $@ $^ $(LDFLAGS)

#=====================================================================

test: $(TEST_PROGS)
	for cmd in $(TEST_PROGS) ; \
	do \
		./$$cmd; \
	done

%.o: %.c
	gcc $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	gcc $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(PROGS) $(TEST_PROGS) vgcore.* core

