SRC=src/
INCLUDE=include
DATABASE=DB2 
MACHINE =LINUX 
WORKLOAD =SSBM 
CFLAGS	= -Wall -Wno-unused-variable  -O2 -DDBNAME=\"dss\" -D$(MACHINE) -D$(DATABASE) -D$(WORKLOAD)
LDFLAGS = -O
# The OBJ,EXE and LIB macros will need to be changed for compilation under
#  Windows NT
OBJ     = .o
LIBS    = -lm 
#
PROG1 = dbgen
PROG2 = qgen
PROGS = $(PROG1) $(PROG2)
#
HDR1 = $(INCLUDE)/dss.h $(INCLUDE)/rnd.h $(INCLUDE)/config.h $(INCLUDE)/dsstypes.h $(INCLUDE)/shared.h $(INCLUDE)/bcd2.h
HDR2 = $(INCLUDE)/tpcd.h $(INCLUDE)/permute.h
HDR  = $(HDR1) $(HDR2)
#
SRC1 = build.c driver.c bm_utils.c rnd.c print.c load_stub.c bcd2.c \
	speed_seed.c text.c permute.c
SRC2 = qgen.c varsub.c 
SRC  = $(SRC1) $(SRC2)
#
OBJ1 = build$(OBJ) driver$(OBJ) bm_utils$(OBJ) rnd$(OBJ) print$(OBJ) \
	load_stub$(OBJ) bcd2$(OBJ) speed_seed$(OBJ) text$(OBJ) permute$(OBJ)
OBJ2 = build$(OBJ) bm_utils$(OBJ) qgen$(OBJ) rnd$(OBJ) varsub$(OBJ) \
	text$(OBJ) bcd2$(OBJ) permute$(OBJ) speed_seed$(OBJ)
OBJS = $(OBJ1) $(OBJ2)
#
SETS = dists.dss 
#
DBGENSRC=$(SRC1) $(HDR1) $(OTHER) $(DOC) $(SRC2) $(HDR2) $(SRC3)
QSRC  = $(FQD) $(VARIANTS)
ALLSRC=$(DBGENSRC) 


%.o: src/%.c $(HDR)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $<  -I$(INCLUDE)
#
all: $(PROGS)
$(PROG1): $(OBJ1) $(SETS) 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ1) $(LIBS) -I$(INCLUDE)
$(PROG2): $(INCLUDE)/permute.h $(OBJ2) 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ2) $(LIBS) -I$(INCLUDE)
clean:
	rm -f $(PROGS) $(OBJS) 
