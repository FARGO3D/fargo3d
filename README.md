# FARGO3D #

#### A versatile MULTIFLUID HD/MHD code that runs on clusters of CPUs or GPUs, with special emphasis on protoplanetary disks. 


![example](https://bytebucket.org/fargo3d/fargo3d_develop/raw/31ab31ea6460ccf6e0bb48ece01d4b60305b4eb9/utils/images/fargo3d.jpg?token=2703b498c71814a39b625f15f9993ee69483b534)

------------------------

##### Website: [fargo.in2p3.fr](http://fargo.in2p3.fr)

##### Documentation can be found [here](https://fargo3d.bitbucket.io/)

##### Clone

```
git clone git@bitbucket.org:fargo3d/fargo3d.git
```

##### Fork & Sync:

Follow the bitbucket [documentation](https://confluence.atlassian.com/bitbucket/forking-a-repository-221449527.html)

##### Working with version 1.3

Switch to the commit ``502c423`` to work with the version 1.3.

```
git checkout -b v1.3 502c423
```

##### Contributing to the code

[Pull requests](https://www.atlassian.com/git/tutorials/making-a-pull-request) are available to the branch ``release/public``. 


------------------------

### First run

#### Sequential CPU

``` 
make SETUP=fargo PARALLEL=0 GPU=0
./fargo3d in/fargo.par
```

#### Parallel CPU

```
make SETUP=fargo PARALLEL=1 GPU=0
mpirun -np 4 ./fargo3d in/fargo.par
```

#### Sequential GPU

```
make SETUP=fargo PARALLEL=0 GPU=1
./fargo3d in/fargo.par
```

#### Parallel GPU

```
make SETUP=fargo PARALLEL=1 GPU=1
mpirun -np 2 ./fargo3d in/fargo.par
```

------------------------

### Description of subdirectories:

* planets: library of planetary systems.

* scripts: python scripts needed to build the code.

* setups: this is where all the custom setup definition are stored. The name of the setups correspond to the names of the directories found in setups/

* src: this is where all the source files of the code are found. Some of them may be redefined in the setups/ subdirectory (the makefile uses the VPATH variable, which behaves much like the PATH variable of the shell, as it allows to decide in which order a given source file is sought within different directories).

* std: this is where all the files that contain some standard definitions (everything that is not   a source file, not a script, and that the user is not supposed to modify). This includes, for   instance, the definition of the some standard boundary conditions, the units (scaling rules) of   the code parameters, etc.

* test_suite: contains python scripts that are used to test various features of the code. They are invoked as follows. We take the example of the permut.py script (which tests that the output of the Orszag-Tang vortex is independent of the choice of coordinates, XY, YZ or XZ). In the main directory (parent directory of test_suite/), simply issue: make testpermut The rule is therefore simply to issue: make test[name of python script without extension] for any script found in this subdirectory. All these scripts should use the 'test' python module found in scripts/

* utils: contains some utilities to post-process the data.
