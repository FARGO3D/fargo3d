This is the main directory of the FARGO3D code (http://fargo.in2p3.fr)

NOTE:
In order to build the code, you must issue a 'make' command in the directory in which this
README file is located. Even though the src/ directory contains a makefile, the latter cannot
be invoked directly. To get started issue simply 'make' for a default built, or 'make help'
to have a description of the most common build options.

Description of subdirectories:

bin: this is where all the intermediate, temporary files of the build will be written (.o object files,
.cu files produced by the C2CUDA parser, etc.)

outputs: directory where the outputs are written by default.

planets: library of planetary systems.

scripts: this is where all python scripts necessary for the build are stored, as well as some
other important scripts.

setups: this is where all the custom setup definition are stored. The name of the setups are
simply the names of the directories found in setups/

src: this is where all the source files of the code are found. Some of them may be redefined
in the setups/ subdirectory (the makefile uses the VPATH variable, which behaves much like
the PATH variable of the shell, as it allows to decide in which order a given source file
is sought within different directories).

std: this is where all the files that contain some standard definitions (everything that is not
a source file, not a script, and that the user is not supposed to modify). This includes, for
instance, the definition of the some standard boundary conditions, the units (scaling rules) of
the code parameters, etc.

test_suite: contains python scripts that are used to test various features of the code.
They are invoked as follows. We take the example of the permut.py script (which tests
that the output of the Orszag-Tang vortex is independent of the choice of coordinates,
XY, YZ or XZ). In the main directory (parent directory of test_suite/), simply issue:
make testpermut
The rule is therefore simply to issue:
make test[name of python script without extension]
for any script found in this subdirectory. All these scripts should use the 'test' python module
found in scripts/

utils: contains some utilities to post-process the data.

For more information, please consult http://fargo.in2p3.fr
