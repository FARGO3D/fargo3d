# FARGO3D #

#### A versatile MULTIFLUID HD/MHD code that runs on clusters of CPUs or GPUs, with special emphasis on protoplanetary disks. 


![example](https://bytebucket.org/fargo3d/public/raw/81b497fb327e916d1c2ad650fe1177b1bbbcc1de/utils/images/fargo3d.jpg)
------------------------

##### Website: [fargo.in2p3.fr](http://fargo.in2p3.fr)

##### [Documentation](https://fargo3d.bitbucket.io/)

------------------------

##### Clone

HTTPS: ```git clone https://bitbucket.org/fargo3d/public.git```

SSH (bitbucket user required): ```git clone git@bitbucket.org:fargo3d/public.git```

##### Fork & Sync:

Follow the bitbucket documentation [here](https://confluence.atlassian.com/bitbucket/forking-a-repository-221449527.html)

##### Working with different versions

Version 2.0: Switch to the commit tagged as ``2.0`` to work with version 2.0.

```
git checkout -b branch2.0 2.0
```

Version 1.3: Switch to the commit tagged as ``1.3`` to work with version 1.3.

```
git checkout -b branch1.3 1.3
```

##### Contributing to the code

[Pull requests](https://www.atlassian.com/git/tutorials/making-a-pull-request) are available to the branch ``release/public``. 

Bugs can be reported to the [issues section](https://bitbucket.org/fargo3d/fargo3d/issues) of the repository or to the [Google group](https://groups.google.com/forum/#!forum/fargo3d).

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