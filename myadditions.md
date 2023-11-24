# COMPILING OPTIONS to be added to .opt

* FARGO_OPT += -DSTOCKHOLMAVV: works only if STOCKHOLM is also defined and applies the damping conditions relaxing to the azimuthal average rather then the intial condition. It is applied to density and radial velocity.
* FARGO_OPT += -DMANUALDAMPBOUNDY: let the boundaries of the damping zone be maually defined as parameters in the par file. Remember to add these parameters to the template parameter file, too!
New params are YDampInf and YDampSup. Damping zones will then be ymin-ydampinf and ydampsup-ymax
* FARGO_OPT += -NOVXSTOCKHOLM: removes the wave killing boundary conditions to the azimuthal velocity