SCRIPTSDIR=scripts
TESTSDIR=test_suite

ifneq (, $(shell which python3))
	PYTHON := $(shell which python3)
else
	PYTHON := $(shell which python2)
endif

all:
	@-${PYTHON} ${SCRIPTSDIR}/make.py ${MAKEFLAGS}
help:
	@echo "\n=================================="
	@echo "Basic options of the build process"
	@echo "=================================="
	@echo "\n* 'make' alone builds the code, while retaining the previous build options provided explicitly (such as the value of the SETUP, GPU=0 or 1, PARALLEL= 0 or 1, etc.)"
	@echo "\n* 'make info' gives the list of build options and their present values"
	@echo "\n* 'make list' gives a list of available setups"
	@echo "\n* 'make SETUP=[setupname] builds the code for a given setup (its value is retained for future builts, until explicitly changed -sticky option- )"
	@echo "\n* 'make clean' removes all intermediate files in order to force a full rebuild, but it does not reset the value of the sticky options (SETUP, PARALLEL, etc.)"
	@echo "\n* 'make mrproper' performs a 'make clean', and it also resets all sticky options. The default setup is 'fargo', which is a 2D polar mesh HD only setup, as for the ancestor code FARGO."
	@echo "\n* Example: to perform a CPU, parallel built of the setup 'mri' issue:\n  'make SETUP=mri PARALLEL=1'"
	@echo "\n* Example: to perform a GPU, serial built of the setup 'fargo' issue:\n  'make SETUP=fargo PARALLEL=0 GPU=1'"
	@echo "\nWarning: in case of trouble with the build, as some dependencies may be broken, try to issue 'make mrproper' and redo the build by specifying explicitly all options."
clean:
	@-${PYTHON} ${SCRIPTSDIR}/make.py clean
	@-rm -f src/rescale.c
	@-rm -fr arch/
mrproper:
	@-${PYTHON} ${SCRIPTSDIR}/make.py clean
	@-rm -f std/.lastflags*
	@-rm -f src/rescale.c
	@-rm -f bin/*
	@-rm -f fargo3d
	@-rm -f scripts/ymin_bound.c
	@-rm -f scripts/ymax_bound.c
	@-rm -f scripts/zmin_bound.c
	@-rm -f scripts/zmax_bound.c
	@-rm -f src/var.c
	@-rm -f src/global_ex.h
	@-rm -f src/param.h
	@-rm -f src/param_noex.h
	@-rm -fr bin/
	@-rm -fr arch/
	@echo "Even the build configuration file was reset"
	@echo "Sticky build options were forgotten"
	@echo
list:
	@echo ""
	@echo "Here is a list of the setups implemented:"
	@echo ""
	@ls setups
	@echo ""
	@echo "Use 'make describe SETUP=setup-name' to get information about 'setup-name'"
describe:
	@echo ""
	@echo "DESCRIPTION OF SETUP "$(SETUP)":"
	@echo ""
	@sed -n '/BEGIN/,/END/p' setups/$(SETUP)/$(SETUP).opt | grep -v BEGIN |grep -v END|sed -e 's/^#//'
	@echo ""
info:
	@echo ""
	@echo "Current sticky build options:"
	@echo ""
ifeq ($(wildcard std/.lastflags),) 
	@cat std/defaultflags
else 	
	@cat std/.lastflags
endif
	@echo ""
save:
	@echo ""
	@echo "Saving present build options for future use"
	@echo "Use 'make restore' to recover them later"
	@echo ""
ifeq ($(wildcard std/.lastflags),) 
	@cp std/defaultflags std/.savebuild
else 	
	@cp std/.lastflags std/.savebuild
endif
restore:
ifeq ($(wildcard std/.savebuild),) 
	@echo "You never saved your build options with 'make save'"
	@echo "I cannot proceed"
else 	
	@echo ""
	@echo "Restoring build options previously saved and rebuilding"
	@echo ""
	@cp std/.savebuild std/.lastflags 
	@make
endif
##Shortcut rules
blocks:
#Syntax: make blocks setup=SETUPNAME
	@echo "Initial built in progress. Please wait"
	@${PYTHON} ${SCRIPTSDIR}/blocks.py --setup=${setup}
cuda:
	@${PYTHON} ${SCRIPTSDIR}/make.py GPU=1 ${MAKEFLAGS}
nocuda:
	@${PYTHON} ${SCRIPTSDIR}/make.py GPU=0 ${MAKEFLAGS}
bigmem:
	@${PYTHON} ${SCRIPTSDIR}/make.py BIGMEM=1 ${MAKEFLAGS}
nobigmem:
	@${PYTHON} ${SCRIPTSDIR}/make.py BIGMEM=0 ${MAKEFLAGS}
seq:
	@${PYTHON} ${SCRIPTSDIR}/make.py PARALLEL=0 ${MAKEFLAGS}
para:
	@${PYTHON} ${SCRIPTSDIR}/make.py PARALLEL=1 ${MAKEFLAGS}
gpu:
	@${PYTHON} ${SCRIPTSDIR}/make.py GPU=1 ${MAKEFLAGS}
nogpu:
	@${PYTHON} ${SCRIPTSDIR}/make.py GPU=0 ${MAKEFLAGS}
noghostsx:
	@${PYTHON} ${SCRIPTSDIR}/make.py GHOSTSX=0 ${MAKEFLAGS}
ghostsx:
	@${PYTHON} ${SCRIPTSDIR}/make.py GHOSTSX=1 ${MAKEFLAGS}
nodebug:
	@${PYTHON} ${SCRIPTSDIR}/make.py DEBUG=0 ${MAKEFLAGS}
debug:
	@${PYTHON} ${SCRIPTSDIR}/make.py DEBUG=1 ${MAKEFLAGS}
nofulldebug:
	@${PYTHON} ${SCRIPTSDIR}/make.py FULLDEBUG=0 ${MAKEFLAGS}
fulldebug:
	@${PYTHON} ${SCRIPTSDIR}/make.py FULLDEBUG=1 ${MAKEFLAGS}
prof:
	@${PYTHON} ${SCRIPTSDIR}/make.py PROFILING=1 ${MAKEFLAGS}
noprof:
	@${PYTHON} ${SCRIPTSDIR}/make.py PROFILING=0 ${MAKEFLAGS}
mpicuda:
	@${PYTHON} ${SCRIPTSDIR}/make.py MPICUDA=1 ${MAKEFLAGS}
nompicuda:
	@${PYTHON} ${SCRIPTSDIR}/make.py MPICUDA=0 ${MAKEFLAGS}
view:
	@${PYTHON} ${SCRIPTSDIR}/make.py FARGO_DISPLAY=MATPLOTLIB ${MAKEFLAGS}
noview:
	@${PYTHON} ${SCRIPTSDIR}/make.py FARGO_DISPLAY=NONE ${MAKEFLAGS}
cgs:
	@${PYTHON} ${SCRIPTSDIR}/make.py UNITS=CGS ${MAKEFLAGS}
mks:
	@${PYTHON} ${SCRIPTSDIR}/make.py UNITS=MKS ${MAKEFLAGS}
scalefree:
	@${PYTHON} ${SCRIPTSDIR}/make.py UNITS=0 ${MAKEFLAGS}
rescale:
	@${PYTHON} ${SCRIPTSDIR}/make.py RESCALE=1 ${MAKEFLAGS}
norescale:
	@${PYTHON} ${SCRIPTSDIR}/make.py RESCALE=0 ${MAKEFLAGS}
longsummary:
	@${PYTHON} ${SCRIPTSDIR}/make.py LONGSUMMARY=1 ${MAKEFLAGS}
shortsummary:
	@${PYTHON} ${SCRIPTSDIR}/make.py LONGSUMMARY=0 ${MAKEFLAGS}
###TEST-SUITE
testlist:
	@ls test_suite/ | sed -e "s/.py//" | grep -v "~"
test%:
	@${PYTHON} ${TESTSDIR}/$*.py
