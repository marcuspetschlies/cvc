CC=gcc
CFLAGS= -c -g -Wall	
LDFLAGS= -lm
SOURCES=average_pi_charm.c stat5.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=average_pi_charm

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@


