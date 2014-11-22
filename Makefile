# Build Process for OpenCL program 

CC = gcc
CPP = g++
CCFLAGS = -g -o3
RM = rm

LIB = -l OpenCL

OBJECTS = \
		  main.o \
		  check_prog.o

PROJ = main.cpp
EXEC = exec



.PHONY: all
.PHONY: clean
.PHONY: compile
.PHONY: acknowledge


#INC_DIRS = $(OCL_INC)
#LIB_DIRS = $(OCL_LIB)

ifdef AMDAPPSDKROOT
 INC_DIRS = $(AMDAPPSDKROOT)/include
 LIB_DIRS = $(AMDAPPSDKROOT)/lib/x86_64
endif

all: $(OBJECTS) compile acknowledge

main.o : main.cpp
	$(CPP) $(CCFLAGS) -o $@ -c $^ -I$(INC_DIRS) 

check_prog.o : check_prog.cpp
	$(CPP) $(CCFLAGS) -o $@ -c $^

compile: $(EXEC)

$(EXEC) : $(OBJECTS)
	$(CPP) $(CCFLAGS) -o $@ $^ -L$(LIB_DIRS) $(LIB)

clean:
	$(RM) $(EXEC) $(OBJECTS)

acknowledge:
		@echo " "
		@echo "Compilation Done Successfully"
		@echo " "
