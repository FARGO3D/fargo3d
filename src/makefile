# If you want your environment variables to have precedence
# you must execute the makefile with the flag "-e" (e.g.: make -e).
#------------------------OPTIONS FILE---------------------------------

SETUPSDIR=../setups
BINDIR=../bin
STDDIR=../std
SRCDIR=../src
ARCHDIR=../arch
ARCHNAME = source.tar.bz2
SCRIPTSDIR=../scripts
GIT_VERSION := $(shell git describe --abbrev=8 --dirty --always --tags 2>/dev/null)
ARCH_CREATE := $(shell mkdir -p $(ARCHDIR))
SHORTARCH := $(shell echo $(ARCHDIR) | sed -e 's|../||')

#------------------------OPTIONS FILE---------------------------------

BLACK=\033[0;30m
BLUE=\033[1;34m #  \e[m'
GREEN=\033[1;32m
CYAN=\033[1;36m
RED=\033[1;31m
PURPLE=\033[1;35m
BROWN=\033[0;33m
LIGHT_GRAY=\033[0;37m
END_COLOR=\033[0m

ifdef FARGO_COLOR
	FARGO_OPT += -DFARGO_COLOR
endif

ifeq (${BIGMEM},1)
	FARGO_OPT += -DBIGMEM
endif
ifeq (${LONGSUMMARY},1)
	FARGO_OPT += -DLONGSUMMARY
endif
ifeq (${RESCALE}, 1)
	FARGO_OPT += -DRESCALE
endif
ifeq (${UNITS}, MKS)
	FARGO_OPT += -DMKS
endif
ifeq (${UNITS}, CGS)
	FARGO_OPT += -DCGS
endif
ifeq (${GHOSTSX}, 1)
	FARGO_OPT += -DGHOSTSX
endif
ifeq (${MPICUDA}, 1)
	FARGO_OPT += -DMPICUDA -DPARALLEL
endif
ifeq (${PROFILING}, 1)
	FARGO_OPT += -DPROFILING
endif
ifeq (${PARALLEL}, 1)
	FARGO_OPT += -DPARALLEL
endif
ifeq (${DEBUG}, 1)
	FARGO_OPT += -DDEBUG
	FARGO_DEBUG = 1
endif
ifeq (${FULLDEBUG}, 1)
	FARGO_OPT += -DDEBUG
	FARGO_OPT += -DWRITEGHOSTS
	FARGO_DEBUG = 1
endif
ifeq (${BLOCKS}, 1)
	FARGO_OPT += -DNOOUTPUTS
endif
ifeq (${GPU}, 1)
	FARGO_OPT += -DGPU
endif
ifdef SETUP
	FARGO_OPT_FILE = ${SETUPSDIR}/${SETUP}/${SETUP}.opt
	VARFILE = ${SETUPSDIR}/${SETUP}/${SETUP}.var
	PARFILE = ${SETUPSDIR}/${SETUP}/${SETUP}.par
endif
ifeq (${NOFLUSH}, 1)
	FARGO_OPT += -DNOFLUSH
endif
ifndef FARGO_OPT_FILE
FARGO_OPT_FILE = ${SETUPSDIR}/fargo/fargo.opt
VARFILE = ${SETUPSDIR}/fargo/fargo.var
endif
ifndef FARGO_DISPLAY
FARGO_DISPLAY = NONE
endif
#---------------------------------------------------------------------
include ${FARGO_OPT_FILE} #The name must not contain digits
#--------------------FARGO_VARIABLES----------------------------------
ifndef FARGO_ARCH
FARGO_ARCH  = LINUX
endif
ifeq (-DGPU,${findstring -DGPU,${FARGO_OPT}})
	FARGO_GPU   = 1
else
	FARGO_GPU   = 0
endif
ifeq (-DDEBUG,${findstring -DDEBUG,${FARGO_OPT}})
	FARGO_DEBUG = 1
endif
#----------------------SYSTEM VARIABLES-------------------------------
ifndef CUDA
CUDA           = /usr/local/cuda
endif
ifndef PYTHON_INCLUDE
PYTHON_INCLUDE = /usr/include/python2.7
endif
#---------------------------------------------------------------------
#LINUX PLATFORM (GENERIC, default)
#FARGO_ARCH must be set to LINUX
CC_LINUX      = gcc 
SEQOPT_LINUX  = -O3
PARAOPT_LINUX = ${SEQOPT_LINUX}
PARACC_LINUX  = mpicc
LIBS_LINUX    = -lm
INC_LINUX     = 
NVCC_LINUX    = nvcc
CUDAOPT_LINUX = -O3 -w #-arch=sm_20 
PARAINC_LINUX = 
PARALIB_LINUX =
CUDAINC_LINUX = -I${CUDA}/include/
CUDALIB_LINUX = -L${CUDA}/lib64/  -lcudart
CUDALIB_LINUX = -L${CUDA}/lib64/  -lcudart -lstdc++
CXX_LINUX     = g++ -stdlib=libstdc++
CXXPARA_LINUX = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#KEPLER PLATFORM WITH OPENMPI-1.7.4 AT UNAM
#FARGO_ARCH must be set to KEPLEROPENMPI
CC_KEPLEROPENMPI      = gcc 
SEQOPT_KEPLEROPENMPI  = -O3 -ffast-math
PARAOPT_KEPLEROPENMPI = ${SEQOPT_KEPLEROPENMPI}
PARACC_KEPLEROPENMPI  = mpicc
LIBS_KEPLEROPENMPI    = -lm
ENVRANK_KEPLEROPENMPI = OMPI_COMM_WORLD_LOCAL_RANK
INC_KEPLEROPENMPI     = 
NVCC_KEPLEROPENMPI    = nvcc
CUDAOPT_KEPLEROPENMPI = -O3 -arch=sm_35
PARAINC_KEPLEROPENMPI = 
PARALIB_KEPLEROPENMPI =
CUDAINC_KEPLEROPENMPI = -I${CUDA}/include/
CUDALIB_KEPLEROPENMPI = -L${CUDA}/lib64/  -lcudart
CXX_KEPLEROPENMPI     = g++ -stdlib=libstdc++
CXXPARA_KEPLEROPENMPI = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#KEPLER PLATFORM WITH MVAPICH2-2.0b AT UNAM
#FARGO_ARCH must be set to KEPLERMVAPICH
CC_KEPLERMVAPICH      = gcc 
SEQOPT_KEPLERMVAPICH  = -O3 -ffast-math
PARAOPT_KEPLERMVAPICH = ${SEQOPT_KEPLERMVAPICH}
PARACC_KEPLERMVAPICH  = mpicc
LIBS_KEPLERMVAPICH    = -lm
ENVRANK_KEPLERMVAPICH = MV2_COMM_WORLD_LOCAL_RANK
INC_KEPLERMVAPICH     = 
NVCC_KEPLERMVAPICH    = nvcc
CUDAOPT_KEPLERMVAPICH = -O3 -arch=sm_35
PARAINC_KEPLERMVAPICH = 
PARALIB_KEPLERMVAPICH = 
CUDAINC_KEPLERMVAPICH = -I${CUDA}/include/
CUDALIB_KEPLERMVAPICH = -L${CUDA}/lib64/ -lcudart -L/share/apps/mvapich2/lib
CXX_KEPLERMVAPICH     = g++ -stdlib=libstdc++
CXXPARA_KEPLERMVAPICH = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#TESLA PLATFORM WITH OPENMPI-1.7.2 AT UNAM
#FARGO_ARCH must be set to TESLAOPENMPI
CC_TESLAOPENMPI      = gcc 
SEQOPT_TESLAOPENMPI  = -O3 -ffast-math
PARAOPT_TESLAOPENMPI = ${SEQOPT_TESLAOPENMPI}
PARACC_TESLAOPENMPI  = mpicc
LIBS_TESLAOPENMPI    = -lm
ENVRANK_TESLAOPENMPI = OMPI_COMM_WORLD_LOCAL_RANK
INC_TESLAOPENMPI     = 
NVCC_TESLAOPENMPI    = nvcc
CUDAOPT_TESLAOPENMPI = -O3 -arch=sm_20
PARAINC_TESLAOPENMPI = 
PARALIB_TESLAOPENMPI =
CUDAINC_TESLAOPENMPI = -I${CUDA}/include/
CUDALIB_TESLAOPENMPI = -L${CUDA}/lib64/  -lcudart
CXX_TESLAOPENMPI     = g++
CXXPARA_TESLAOPENMPI = mpic++
#---------------------------------------------------------------------
#TESLA PLATFORM WITH MVAPICH2-1.9 AT UNAM
#FARGO_ARCH must be set to TESLAMVAPICH
CC_TESLAMVAPICH      = gcc 
SEQOPT_TESLAMVAPICH  = -O3 -ffast-math
PARAOPT_TESLAMVAPICH = ${SEQOPT_TESLAMVAPICH}
PARACC_TESLAMVAPICH  = mpicc
LIBS_TESLAMVAPICH    = -lm
ENVRANK_TESLAMVAPICH = MV2_COMM_WORLD_LOCAL_RANK
INC_TESLAMVAPICH     = 
NVCC_TESLAMVAPICH    = nvcc
CUDAOPT_TESLAMVAPICH = -O3 -arch=sm_20
PARAINC_TESLAMVAPICH = 
PARALIB_TESLAMVAPICH = -lmpich -lOpenCL
CUDAINC_TESLAMVAPICH = -I${CUDA}/include/
CUDALIB_TESLAMVAPICH = -lcudart
CXX_TESLAMVAPICH     = g++ -stdlib=libstdc++
CXXPARA_TESLAMVAPICH = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#TESLA C2070 PLATFORM WITH MVAPICH2-1.9 AT CORDOBA
#FARGO_ARCH must be set to MENDIETA
CC_MENDIETA      = gcc 
SEQOPT_MENDIETA  = -O3 -ffast-math
PARAOPT_MENDIETA = ${SEQOPT_MENDIETA}
PARACC_MENDIETA  = mpicc
LIBS_MENDIETA    = -lm
ENVRANK_MENDIETA = MV2_COMM_WORLD_RANK
INC_MENDIETA     = 
NVCC_MENDIETA    = nvcc
CUDAOPT_MENDIETA = -O3 -arch=sm_20
PARAINC_MENDIETA = 
PARALIB_MENDIETA = -lmpich -lOpenCL
CUDAINC_MENDIETA = -I${CUDA}/include/
CUDALIB_MENDIETA = -lcudart
CXX_MENDIETA     = g++ -stdlib=libstdc++
CXXPARA_MENDIETA = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#QUADRO PLATFORM (IATE COMPUTER)
#FARGO_ARCH must be set to QUADRO
CC_QUADRO      = gcc
SEQOPT_QUADRO  = -O3 #-ffast-math 
PARAOPT_QUADRO = ${SEQOPT_QUADRO}
PARACC_QUADRO  = mpicc
LIBS_QUADRO    = -lm
INC_QUADRO     = 
NVCC_QUADRO    = ${CUDA}/bin/nvcc -arch=sm_20
CUDAOPT_QUADRO = 
PARAINC_QUADRO = -I/usr/include/openmpi-x86_64/
PARALIB_QUADRO = -L/usr/lib64/openmpi/lib/
CUDALIB_QUADRO = -L${CUDA}/lib64/ -lcudart
CUDAINC_QUADRO = -I${CUDA}/include/
CXX_QUADRO     = g++ -stdlib=libstdc++
CXXPARA_QUADRO = mpic++ -stdlib=libstdc++
#---------------------------------------------------------------------
#MAC PLATFORM (FREDERIC'S COMPUTER) - Works with OSX 10.10 (Yosemite)
#and CUDA release 7.5, V7.5.19 + GPU Driver Version: 10.4.2 310.41.35f01
#FARGO_ARCH must be set to MacIntel
CC_MacIntel      = gcc
SEQOPT_MacIntel  = -O3 -arch x86_64 -Wmissing-prototypes #-Wall -Wextra
PARAOPT_MacIntel = ${SEQOPT_MacIntel}
PARACC_MacIntel  = mpicc
ENVRANK_MacIntel = OMPI_COMM_WORLD_RANK
LIBS_MacIntel    = 
INC_MacIntel     = 
NVCC_MacIntel    = ${CUDA}/bin/nvcc
CUDAOPT_MacIntel = -arch=sm_30 -m64 -O3 --compiler-options '-Wno-return-type-c-linkage' -Xcudafe --diag_suppress=set_but_not_used -Xcudafe --diag_suppress=declared_but_not_referenced
PARAINC_MacIntel = 
PARALIB_MacIntel = 
CUDALIB_MacIntel = -L${CUDA}/lib/ -lcudart
CUDAINC_MacIntel = -I${CUDA}/include/
CXX_MacIntel     = g++ -stdlib=libstdc++
CXXPARA_MacIntel = mpic++ -stdlib=libstdc++
#--------------Architecture independent names-------------------------
CC	 = ${CC_${FARGO_ARCH}}
SEQOPT   = ${SEQOPT_${FARGO_ARCH}}
PARAOPT  = ${PARAOPT_${FARGO_ARCH}}
PARACC   = ${PARACC_${FARGO_ARCH}}
LIBS     = ${LIBS_${FARGO_ARCH}}
NVCC     = ${NVCC_${FARGO_ARCH}}
ENVRANK  = ${ENVRANK_${FARGO_ARCH}}
CUDAOPT  = ${CUDAOPT_${FARGO_ARCH}}
PARAINC  = ${PARAINC_${FARGO_ARCH}}
PARALIB  = ${PARALIB_${FARGO_ARCH}}
CUDAINC  = ${CUDAINC_${FARGO_ARCH}}
CUDALIB  = ${CUDALIB_${FARGO_ARCH}}
CXX      = ${CXX_${FARGO_ARCH}}
CXXPARA  = ${CXXPARA_${FARGO_ARCH}}
INCLUDE  = ${INC_${FARGO_ARCH}} ${CUDAINC_${FARGO_ARCH}} \
	  -I${SRCDIR} -I${BINDIR}
#---------------------------------------------------------------------

MAKEFILE= makefile

MAKEFILE += ${SCRIPTSDIR}/make.py

GLOBAL   = ${MAKEFILE} ${SCRIPTSDIR}/c2cuda.py ${FARGO_OPT_FILE}

GLOBAL += fargo3d.h define.h types_def.h prototypes.h fondam.h structs.h

BOUNDARIES_DEP = boundparser.py boundary_template.c  \
	    	 $(wildcard ${SETUPSDIR}/${SETUP}/$(SETUP).bound.*)




MPIDUMMY = mpi_dummy.o #dummy file, for sequential runs.
EXENAME  = ../fargo3d  #Name and location of the executable file


VPATH=${SETUPSDIR}/${SETUP}:${STDDIR}:${SCRIPTSDIR}:${SRCDIR}

#---------------#Sequential and parallel objects----------------------
MAINOBJ  = LowTasks.o psys.o main.o param.o var.o usage.o   	    \
	   split.o comm.o CompPresIso.o CompPresPoly.o         	    \
	   boundary.o checknans.o 				    \
	   vanleer_x_a.o vanleer_x_b.o 				    \
           vanleer_y_a.o vanleer_y_b.o				    \
	   vanleer_z_a.o vanleer_z_b.o 				    \
	   divrho.o transport.o 				    \
	   substep3.o CompPresAd.o			    	    \
	   substep2_a.o substep2_b.o condinit.o     		    \
	   output.o output2d.o output_vtk.o 			    \
	   output_vtk_merging.o			                    \
	   algogas.o planets.o runge_kutta.o	        	    \
	   stockholm.o mhd.o potential.o	           	    \
	   timeinfo.o allocate.o              		      	    \
	   defout.o fresh.o  light_global_dev.o          	    \
	   fargo_mhd.o substep1_x.o substep1_y.o substep1_z.o	    \
	   compfields.o advect_shift.o rescale.o		    \
	   momenta_x.o momenta_y.o momenta_z.o			    \
	   reduction_sum.o reduction_min.o			    \
	   update_x.o update_y.o update_z.o 			    \
           update_density_x.o update_density_y.o update_density_z.o \
	   newvel_x.o newvel_y.o newvel_z.o 			\
	   compute_residual.o change_frame.o compute_vmed.o	\
           correct_vtheta.o cfl.o cfl_b.o			\
	   compute_force.o compute_slopes.o compute_star.o	\
	   compute_emf.o update_magnetic.o lorentz_force.o	\
	   fargo_ppa_a.o fargo_ppa_b.o fargo_ppa_c.o            \
	   fargo_ppa_d.o fargo_ppa_d_2d.o fargo_ppa_steep.o	\
	   vanleerx_ppa.o dh_boundary.o cpvel.o         	\
	   integrate_emf.o monitor.o resist.o resistivity.o	\
	   viscosity.o init_stockholm.o visctensor_cart.o	\
	   addviscosity_cart.o addviscosity_cyl.o		\
	   addviscosity_sph.o visctensor_sph.o			\
	   visctensor_cyl.o stretch.o 				\
	   mon_dens.o mon_momx.o mon_momy.o mon_momz.o          \
	   mon_torque.o comm_device.o checkmute_y.o		\
	   checkmute_z.o mon_reynolds.o mon_maxwell.o           \
	   mon_bxflux.o                                         \
	   postrestarthook.o change_arch.o select_device.o	\
	   redefoptions.o setuphook1.o				\
           fill_ghostsx.o cfl_fluids_min.o colrate.o 		\
	   collisions.o resetfield.o totaldens.o		\
	   floor.o ShearingPeriodicBC.o summary.o		\
	   dust_diffusion_main.o dust_diffusion_core.o		\
	   dust_diffusion_coefficients.o copy_field.o
#-----------------------#Cuda objects---------------------------------
GPU_OBJ  =  reduction_sum_device.o reduction_min_device.o	\
	   	lowtasks_gpu.o 


#<CUDA OBJECTS> #Do not remove!!!
GPU_OBJBLOCKS =  CompPresIso_gpu.o CompPresAd_gpu.o 		\
	   compute_slopes_gpu.o compute_star_gpu.o		\
	   compute_emf_gpu.o update_magnetic_gpu.o		\
	   substep1_x_gpu.o substep1_y_gpu.o substep1_z_gpu.o 	\
           substep2_a_gpu.o substep2_b_gpu.o substep3_gpu.o 	\
	   divrho_gpu.o stockholm_gpu.o 			\
	   vanleer_x_a_gpu.o vanleer_x_b_gpu.o 			\
	   vanleer_y_a_gpu.o vanleer_y_b_gpu.o 			\
	   vanleer_z_a_gpu.o vanleer_z_b_gpu.o 			\
	   momenta_x_gpu.o momenta_y_gpu.o momenta_z_gpu.o	\
	   update_x_gpu.o update_y_gpu.o update_z_gpu.o 	\
	   update_density_x_gpu.o update_density_y_gpu.o update_density_z_gpu.o\
           newvel_x_gpu.o newvel_y_gpu.o newvel_z_gpu.o 	\
	   advect_shift_gpu.o compute_residual_gpu.o 		\
	   change_frame_gpu.o potential_gpu.o 			\
	   correct_vtheta_gpu.o cfl_gpu.o compute_force_gpu.o	\
	   lorentz_force_gpu.o CompPresPoly_gpu.o 		\
	   fargo_ppa_a_gpu.o fargo_ppa_b_gpu.o			\
	   fargo_ppa_c_gpu.o fargo_ppa_d_gpu.o			\
	   fargo_ppa_d_2d_gpu.o resist_gpu.o 			\
	   integrate_emf_gpu.o visctensor_cart_gpu.o 		\
	   addviscosity_cart_gpu.o addviscosity_cyl_gpu.o	\
	   visctensor_cyl_gpu.o addviscosity_sph_gpu.o		\
	   mon_dens_gpu.o mon_momx_gpu.o mon_momy_gpu.o         \
	   mon_momz_gpu.o mon_torque_gpu.o checkmute_y_gpu.o	\
	   checkmute_z_gpu.o mon_reynolds_gpu.o                 \
	   mon_maxwell_gpu.o visctensor_sph_gpu.o               \
	   mon_bxflux_gpu.o                   			\
	   fargo_ppa_steep_gpu.o setuphook1_gpu.o		\
           fill_ghostsx_gpu.o collisions_gpu.o                  \
           totaldens_gpu.o floor_gpu.o		                \
	   resetfield_gpu.o dust_diffusion_core_gpu.o		\
	   dust_diffusion_coefficients_gpu.o

#</CUDA OBJECTS>
#---------------------------------------------------------------------

ifeq (-DY,${findstring -DY,${FARGO_OPT}})
MAINOBJ += $(foreach var, $(FLUIDS), ymin_bound_$(var).o)
MAINOBJ += $(foreach var, $(FLUIDS), ymax_bound_$(var).o)
GPU_OBJ += $(foreach var, $(FLUIDS), ymin_bound_$(var)_gpu.o)
GPU_OBJ += $(foreach var, $(FLUIDS), ymax_bound_$(var)_gpu.o)
endif

ifeq (-DZ,${findstring -DZ,${FARGO_OPT}})
MAINOBJ += $(foreach var, $(FLUIDS), zmin_bound_$(var).o)
MAINOBJ += $(foreach var, $(FLUIDS), zmax_bound_$(var).o)
GPU_OBJ += $(foreach var, $(FLUIDS), zmin_bound_$(var)_gpu.o)
GPU_OBJ += $(foreach var, $(FLUIDS), zmax_bound_$(var)_gpu.o)
endif


MAINOBJ += ${CPU_OBJECTS}
GPUOBJ += ${GPU_OBJ} ${GPU_OBJBLOCKS}

-include ${SETUPSDIR}/${SETUP}/${SETUP}.objects

ifeq (-DVISCOUSDRIFT,${findstring -DVISCOUSDRIFT,${FARGO_OPT}})
	LIBS += -lgsl -lgslcblas
endif

ifeq (-DPARALLEL,${findstring -DPARALLEL,${FARGO_OPT}})
	OPTIONS += ${PARAOPT}
	COMPILER = ${PARACC}
	LIBS += ${PARALIB}
else
	OPTIONS  += ${SEQOPT}
	MAINOBJ += ${MPIDUMMY}
	COMPILER = ${CC}
endif
	LINKER = ${COMPILER}


OBJECTS = ${MAINOBJ}

ifeq (${FARGO_GPU}, 1)
	OBJECTS += ${GPUOBJ}
	LIBS += ${CUDALIB}
endif

#---------------------------------------------------------------------
#Adding includes and objects if MATPLOTLIB is activated.
#These lines prevent compatibility problems 
#if matplotlib is not installed
ifeq (MATPLOTLIB,${findstring MATPLOTLIB,${FARGO_DISPLAY}})
	FARGO_OPT += -DMATPLOTLIB
	INCLUDE   += -I${PYTHON_INCLUDE}
	OBJECTS   +=  matplotlib.o
	LIBS      += -lpython2.7
endif
# Debug mode if FARGO_DEBUG = 1, Note that this mode is only 
# meant to execute gdb (or lldb, etc.) or valgrind.
ifeq (${FARGO_DEBUG}, 1)
OPTIONS = -g
endif

ifeq (${MPICUDA}, 1)
	FARGO_OPT += -DENVRANK='${ENVRANK}'
endif

all:
	@echo "You must be in the upper level directory to invoke 'make'"
	@echo "Do 'cd ..' then 'make' to proceed."

allp:  ${EXENAME}
	@echo ""
ifdef FARGO_COLOR
	@echo "${BLUE}"
endif
	@echo "          FARGO3D SUMMARY:           "
	@echo "          ===============            "
	@echo ""
ifeq (-DPARALLEL,${findstring -DPARALLEL,${FARGO_OPT}})
	@echo 'This built is PARALLEL (MPI). Use "make seq" to change that'
ifeq (${FARGO_GPU}, 1)
	@echo ""
	@echo "This built can be launched on"
	@echo "a cluster of GPUs."
endif
else
	@echo 'This built is SEQUENTIAL. Use "make para" to change that'
ifeq (${FARGO_GPU}, 1)
	@echo ""
	@echo "This built can be launched on"
	@echo "a CPU with a GPU card (1 GPU only)."
endif
endif
ifeq (MATPLOTLIB,${findstring MATPLOTLIB,${FARGO_DISPLAY}})	
	@echo ""
	@echo "This built has a graphical output,"
	@echo "which uses Python's matplotlib library."
	@echo 'Use "make noview" to change that.'
endif
	@echo ""
	@echo ""
	@echo "SETUP:      '"${SETUP}"'         "
	@echo '(Use "make SETUP=[valid_setup_string]" to change set up)'
	@echo '(Use "make list" to see the list of setups implemented)'
	@echo '(Use "make info" to see the current sticky build options)'
	@echo ""
ifeq (${PROFILING}, 1)
	@echo ""
	@echo "==================================="
	@echo "|   PROFILING MODE ACTIVATED!     |"
	@echo "==================================="
	@echo '(Use "make noprof" to change that)'
	@echo ""
endif
ifeq (${MPICUDA}, 1)
	@echo ""
	@echo "==================================="
	@echo "|   CUDA AWARE MPI!               |"
	@echo "==================================="
	@echo "Direct GPU-GPU MPI communications"
	@echo '(Use "make nompicuda" to change that)'
	@echo ""
endif
ifeq (${DEBUG}, 1)
	@echo ""
	@echo "==================================="
	@echo "|       DEBUG MODE ACTIVATED!     |"
	@echo "==================================="
	@echo '(Use "make nodebug" to change that)'
	@echo ""
endif
ifeq (${FULLDEBUG}, 1)
	@echo ""
	@echo "==================================="
	@echo "|  FULL DEBUG MODE ACTIVATED!     |"
	@echo "==================================="
	@echo "In FULL DEBUG mode buffer zones (ghosts) are included in output files"
	@echo '(Use "make nofulldebug" to change that)'
	@echo ""
endif
ifdef FARGO_COLOR
	@echo "${END_COLOR}"
endif

#---------------------------------------------------------------------
#------------------------MAIN RULE------------------------------------

${EXENAME} : ${SCRIPTSDIR}/bound_proto.code ${OBJECTS}
	@echo ""
	@echo "All objects are OK. Linking stage"
# You may remove the '@' sign in the instructions below if you need a verbose output
	@${LINKER} ${OBJECTS} -o ${EXENAME} ${LIBS}
	@/bin/cp -f ${SRCDIR}/*.h $(ARCHDIR)
	@/bin/cp -f ${BINDIR}/*.h $(ARCHDIR)
	@cd $(ARCHDIR)/..;/bin/rm -f temparch; tar jcf temparch $(SHORTARCH) 2> /dev/null
	@mv $(ARCHDIR)/../temparch $(ARCHDIR)/$(ARCHNAME)
ifdef LONGSUMMARY
	@cat ${SETUPSDIR}/${SETUP}/${SETUP}.bound.* >> $(EXENAME)
	@cat ${SETUPSDIR}/${SETUP}/${SETUP}.bound.* | wc -c | xargs printf "%012d" >> ${EXENAME}
	@cat $(ARCHDIR)/$(ARCHNAME) >> $(EXENAME)
	@wc -c $(ARCHDIR)/$(ARCHNAME) | sed -e "s|$(ARCHDIR).*$$||" | xargs printf "%012d" >> ${EXENAME}
	@cat $(STDDIR)/.lastflags >> ${EXENAME}
	@wc -c $(STDDIR)/.lastflags | sed -e "s|$(STDDIR).*$$||" | xargs printf "%012d" >> ${EXENAME}
endif
ifdef FARGO_COLOR
	@echo -n "${GREEN}"
	@echo "The executable file has been created."
	@echo -n "${END_COLOR}"
endif
#---------------------------------------------------------------------

var.c: ${PARFILE} stdpar.par #$(VARFILE) default_variables.dat
	@echo ""
	@echo "Generating var.c file"
	@python ${SCRIPTSDIR}/par.py $(PARFILE) ${STDDIR}/stdpar.par

${SETUP}.units:
	@touch ${SETUPSDIR}/${SETUP}/${SETUP}.units

rescale.c: ${PARFILE} ${SCRIPTSDIR}/unitparser.py ${STDDIR}/standard.units ${SETUP}.units
	@echo ""
	@echo "Generating rescale.c file"
	@python ${SCRIPTSDIR}/unitparser.py ${SETUP}

.PHONY: clean mrproper force
gitversion: force
	@echo '$(GIT_VERSION)' | cmp -s - $@ || echo '$(GIT_VERSION)' > $@
clean:
	@-rm ${SCRIPTSDIR}/bound_*.code
	@-rm ${BINDIR}/*
	@echo
	@echo "Everything has been cleaned"
	@echo

main.o: gitversion
change_arch.o : change_arch.c ${SCRIPTSDIR}/bound_cpu.code ${SCRIPTSDIR}/bound_gpu.code
	@echo CC $< "==>" $*.o
	@${COMPILER} $< -c ${INCLUDE} ${OPTIONS} ${CUDAINC} ${FARGO_OPT} -DMONITOR_SCALAR='${MONITOR_SCALAR}' -DMONITOR_2D='${MONITOR_2D}' -DMONITOR_Y='${MONITOR_Y}' -DMONITOR_Y_RAW='${MONITOR_Y_RAW}' -DMONITOR_Z='${MONITOR_Z}' -DMONITOR_Z_RAW='${MONITOR_Z_RAW}' -DSETUPNAME='${SETUP}'

${SCRIPTSDIR}/bound_proto.code ${SCRIPTSDIR}/bound_cpu.code ${SCRIPTSDIR}/bound_gpu.code:
	@echo "Generating bound_code files..."
	@python ${SCRIPTSDIR}/bound_code.py ${SETUP} ${NFLUIDS}

collisions.cu: collisions_template.cu ${GLOBAL}
	@echo "PARSING collisions.cu"
	@python ${SCRIPTSDIR}/collisions_gpu.py ${NFLUIDS}

%_gpu.o : %.cu ${GLOBAL} #Rule: test_gpu.o is created from test.cu
	@echo NVCC $< "==>" $*_gpu.o
	@/bin/cp -f $< $(ARCHDIR)
	@${NVCC} $< -c ${CUDAOPT} ${CUDAINC} ${INCLUDE} -o $*_gpu.o ${FARGO_OPT}

%.cu: %.c
	@echo C2CUDA $< "==>" $*.cu
	@python ${SCRIPTSDIR}/c2cuda.py -i $< -o $*.cu -s ${SETUP}	

reduction_sum_device.o: reduction_sum_device.cu
	@echo NVCC $< "==>" $*.o
	@/bin/cp -f $< $(ARCHDIR)
	@${NVCC} $^ -c ${CUDAOPT} ${CUDAINC} ${INCLUDE}	${FARGO_OPT}

reduction_min_device.o: reduction_min_device.cu
	@echo NVCC $< "==>" $*.o
	@/bin/cp -f $< $(ARCHDIR)
	@${NVCC} $^ -c ${CUDAOPT} ${CUDAINC} ${INCLUDE}	${FARGO_OPT}


# If HARDBOUNDARIES is set in the .opt file, boundaries are not created by the script
ifneq (-DHARDBOUNDARIES,${findstring -DHARDBOUNDARIES,${FARGO_OPT}})
ymin_bound_%.o: ymin_bound_%.c ${GLOBAL}
	@echo CC $< "==>" ymin_bound_$*.o
	@${COMPILER} ymin_bound_$*.c -c ${INCLUDE} ${OPTIONS} ${FARGO_OPT}
ymax_bound_%.o: ymax_bound_%.c ${GLOBAL}
	@echo CC $< "==>" ymax_bound_$*.o
	@${COMPILER} ymax_bound_$*.c -c ${INCLUDE} ${OPTIONS} ${FARGO_OPT}
zmin_bound_%.o: zmin_bound_%.c ${GLOBAL}
	@echo CC $< "==>" zmin_bound_$*.o
	@${COMPILER} zmin_bound_$*.c -c ${INCLUDE} ${OPTIONS} ${FARGO_OPT}
zmax_bound_%.o: zmax_bound_%.c ${GLOBAL}
	@echo CC $< "==>" zmax_bound_$*.o
	@${COMPILER} zmax_bound_$*.c -c ${INCLUDE} ${OPTIONS} ${FARGO_OPT}

ymin_bound_%.c ymax_bound_%.c zmin_bound_%.c zmax_bound_%.c: boundaries.txt centering.txt ${SETUPSDIR}/${SETUP}/$(SETUP).bound.% ${GLOBAL} ${BOUNDARIES_DEP}
	@-python ${SCRIPTSDIR}/boundparser.py $^

endif

# You may remove the '@' sign in the instructions below if you need a verbose output
%.o : %.c ${GLOBAL} ${SCRIPTSDIR}/bound_proto.code
	@echo CC $< "==>" $*.o
	@/bin/cp -f $< $(ARCHDIR)
	@${COMPILER} $< -c ${INCLUDE} ${OPTIONS} ${CUDAINC} ${FARGO_OPT}  	\
	-DMONITOR_SCALAR='${MONITOR_SCALAR}' -DMONITOR_2D='${MONITOR_2D}'	\
	-DMONITOR_Y='${MONITOR_Y}' -DMONITOR_Y_RAW='${MONITOR_Y_RAW}' 		\
	-DMONITOR_Z='${MONITOR_Z}' -DMONITOR_Z_RAW='${MONITOR_Z_RAW}' 		\
	-DSETUPNAME='${SETUP}' -DVERSION=$(GIT_VERSION) 			\
	-DPATHARCH=$(CURDIR)/$(ARCHDIR)/$(ARCHNAME) -DOPTIONS="${FARGO_OPT}"
