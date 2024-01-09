# FARGO3D #

#### A versatile MULTIFLUID HD/MHD code that runs on clusters of CPUs or GPUs, with special emphasis on protoplanetary disks. 

### [Documentation](https://fargo3d.github.io/documentation)

Report bugs to the [issues section](https://github.com/FARGO3D/fargo3d/issues) or to the [Google group](https://groups.google.com/forum/#!forum/fargo3d).

### First run

#### Sequential CPU

``` 
make SETUP=fargo PARALLEL=0 GPU=0
./fargo3d setups/fargo/fargo.par
```

#### Parallel CPU

```
make SETUP=fargo PARALLEL=1 GPU=0
mpirun -np 8 ./fargo3d setups/fargo/fargo.par
```

#### Sequential GPU

```
make SETUP=fargo PARALLEL=0 GPU=1
./fargo3d setups/fargo/fargo.par
```

#### Parallel GPU

```
make SETUP=fargo PARALLEL=1 GPU=1
mpirun -np 2 ./fargo3d setups/fargo/fargo.par
```

------------------------

### Description of subdirectories:

* ```in/```: setup parameter files. Equivalent to the ```.par``` files in the ```setup/``` subdirectory.

* ```planets/```: planets configuration files.

* ```scripts/```: scripts used to build the code.

* ```setups/```: custom setup definitions.

* ```src/```: source files. These files can be copied to the ```setups/``` subdirectory and modified there. The makefile uses the ```VPATH``` variable to decide in which order a given source file is sought within different directories (the ```
setup/``` subdirectory has higher prority than ```src```).

* ```std/```: standard or default definitions. These definitions include standard boundary conditions, units, scaling rules, default setup parameters, etc.

* ```test_suite/```: scripts used to test the code. The rule to issue them is ```make test[name of python script without extension]``` for any script in this subdirectory. These scripts use the 'test' python module in ```
scripts/```

* ```utils/```: utilities to post-process the data.
