set(main_sources main.c LowTasks.c psys.c param.c usage.c split.c comm.c
    CompPresIso.c CompPresPoly.c boundary.c checknans.c vanleer_x_a.c
    vanleer_x_b.c vanleer_y_a.c vanleer_y_b.c vanleer_z_a.c vanleer_z_b.c
    divrho.c transport.c substep3.c CompPresAd.c substep2_a.c substep2_b.c
    output.c output2d.c output_vtk.c output_vtk_merging.c
    algogas.c planets.c runge_kutta.c stockholm.c mhd.c potential.c
    timeinfo.c allocate.c defout.c fresh.c light_global_dev.c fargo_mhd.c
    substep1_x.c substep1_y.c substep1_z.c compfields.c advect_shift.c
    rescale.c momenta_x.c momenta_y.c momenta_z.c reduction_sum.c
    reduction_min.c update_x.c update_y.c update_z.c update_density_x.c
    update_density_y.c update_density_z.c newvel_x.c newvel_y.c newvel_z.c
    compute_residual.c change_frame.c compute_vmed.c correct_vtheta.c cfl.c
    cfl_b.c compute_force.c compute_slopes.c compute_star.c compute_emf.c
    update_magnetic.c lorentz_force.c fargo_ppa_a.c fargo_ppa_b.c fargo_ppa_c.c
    fargo_ppa_d.c fargo_ppa_d_2d.c fargo_ppa_steep.c vanleerx_ppa.c
    dh_boundary.c cpvel.c integrate_emf.c monitor.c viscosity.c init_stockholm.c
    visctensor_cart.c addviscosity_cart.c addviscosity_cyl.c addviscosity_sph.c
    visctensor_sph.c visctensor_cyl.c stretch.c mon_dens.c mon_momx.c mon_momy.c
    mon_momz.c mon_torque.c comm_device.c checkmute_y.c checkmute_z.c
    mon_reynolds.c mon_maxwell.c mon_bxflux.c postrestarthook.c change_arch.c
    select_device.c redefoptions.c setuphook1.c fill_ghostsx.c cfl_fluids_min.c
    colrate.c collisions.c resetfield.c totaldens.c floor.c ShearingPeriodicBC.c
    summary.c dust_diffusion_main.c dust_diffusion_core.c
    dust_diffusion_coefficients.c copy_field.c nimhd_hall_effect.c
    nimhd_hall_effect_emfx.c nimhd_hall_effect_emfy.c nimhd_hall_effect_emfz.c
    nimhd_hall_effect_updateB.c nimhd_hall_effect_update_emfs.c
    nimhd_hall_effect_coeff.c nimhd_ambipolar_diffusion.c
    nimhd_ambipolar_diffusion_emfx.c nimhd_ambipolar_diffusion_emfy.c
    nimhd_ambipolar_diffusion_emfz.c nimhd_ambipolar_diffusion_coeff.c
    nimhd_ohmic_diffusion.c nimhd_ohmic_diffusion_coeff.c
    compute_jx.c compute_jy.c compute_jz.c mesh.c ram_advect.c ram_advect_lin.c
    ram_compute_ustar.c ram_plm.c)

set(std_dir ${CMAKE_SOURCE_DIR}/std)
set(script_dir ${CMAKE_SOURCE_DIR}/scripts)

set(boundary_inputs "")
set(boundary_outputs "")

math(EXPR stop "${NFLUIDS} - 1")
foreach(fluid RANGE ${stop})
    list(APPEND boundary_inputs ${setup_dir}/${SETUP}.bound.${fluid})
    if(YDIM)
        list(APPEND boundary_outputs
            ${CMAKE_CURRENT_BINARY_DIR}/ymin_bound_${fluid}.c
            ${CMAKE_CURRENT_BINARY_DIR}/ymax_bound_${fluid}.c)
    endif()
    if(ZDIM)
        list(APPEND boundary_outputs
            ${CMAKE_CURRENT_BINARY_DIR}/zmin_bound_${fluid}.c
            ${CMAKE_CURRENT_BINARY_DIR}/zmax_bound_${fluid}.c)
    endif()
endforeach()

add_library(fargo3d_generated
    ${CMAKE_CURRENT_BINARY_DIR}/var.c ${boundary_outputs})

find_package(Python REQUIRED COMPONENTS Interpreter)

if(boundary_inputs AND boundary_outputs)
    add_custom_command(OUTPUT ${boundary_outputs} COMMAND ${Python_EXECUTABLE}
        ARGS ${CMAKE_SOURCE_DIR}/scripts/boundparser.py ${boundary_inputs}
        -t ${std_dir}/boundary_template.c -b ${std_dir}/boundaries.txt
        -c ${std_dir}/centering.txt -o ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${script_dir}/boundparser.py ${std_dir}/boundary_template.c
        ${std_dir}/boundaries.txt ${std_dir}/centering.txt ${boundary_inputs})
    add_custom_target(generate_bnd_c DEPENDS ${boundary_outputs})
    add_dependencies(fargo3d_generated generate_bnd_c)
endif()

# compile boundary function prototypes
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bound_cpu.code
    ${CMAKE_CURRENT_BINARY_DIR}/bound_gpu.code ${CMAKE_CURRENT_BINARY_DIR}/bound_proto.code
    COMMAND ${Python_EXECUTABLE} ARGS ${script_dir}/bound_code.py
    -n ${NFLUIDS} -cpu ${CMAKE_CURRENT_BINARY_DIR}/bound_cpu.code
    -gpu ${CMAKE_CURRENT_BINARY_DIR}/bound_gpu.code
    -proto ${CMAKE_CURRENT_BINARY_DIR}/bound_proto.code
    DEPENDS ${script_dir}/bound_code.py
    COMMENT "Compiling bound_cpu.code, bound_gpu.code, and bound_proto.code")
add_custom_target(generate_bnd_h DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/bound_cpu.code
    ${CMAKE_CURRENT_BINARY_DIR}/bound_gpu.code
    ${CMAKE_CURRENT_BINARY_DIR}/bound_proto.code)

# compile .par files -> var.c
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/var.c
    COMMAND ${Python_EXECUTABLE} ARGS ${script_dir}/par.py
    ${setup_dir}/${SETUP}.par ${std_dir}/stdpar.par
    -m ${setup_dir}/${SETUP}.mandatories -o ${CMAKE_CURRENT_BINARY_DIR}/var.c
    DEPENDS ${script_dir}/par.py ${setup_dir}/${SETUP}.par ${std_dir}/stdpar.par
    COMMENT "Compiling .par files -> var.c")
add_custom_target(generate_var_c DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/var.c)

# compile var.c -> param.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/param.h
    COMMAND ${Python_EXECUTABLE} ARGS ${script_dir}/param.py
    ${CMAKE_CURRENT_BINARY_DIR}/var.c -o ${CMAKE_CURRENT_BINARY_DIR}/param.h
    DEPENDS ${script_dir}/param.py generate_var_c
    COMMENT "Compiling var.c -> param.h")
add_custom_target(generate_param_h DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/param.h)

# compile global.h -> global_ex.h
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/global_ex.h
    COMMAND ${Python_EXECUTABLE} ARGS ${script_dir}/global.py
    ${CMAKE_CURRENT_SOURCE_DIR}/global.h -o ${CMAKE_CURRENT_BINARY_DIR}/global_ex.h
    DEPENDS ${script_dir}/global.py ${CMAKE_CURRENT_SOURCE_DIR}/global.h
    COMMENT "Compiling global.h -> global_ex.h")
add_custom_target(generate_global_ex_h DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/global_ex.h)

add_dependencies(fargo3d_generated generate_bnd_h
    generate_var_c generate_param_h generate_global_ex_h)
target_include_directories(fargo3d_generated PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(fargo3d PUBLIC fargo3d_generated)
target_include_directories(fargo3d PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

target_sources(fargo3d PUBLIC ${main_sources})
target_link_libraries(fargo3d PUBLIC m)

# conditionally add MPI dependency or dummy library
if(PARALLEL)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(fargo3d PUBLIC MPI::MPI_C)
    target_link_libraries(fargo3d_generated PUBLIC MPI::MPI_C)
else()
    target_sources(fargo3d PUBLIC mpi_dummy.c)
endif()

# conditionally add GPU support
if(GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(gpu_sources_in CompPresIso.c CompPresAd.c compute_slopes.c
        compute_star.c compute_emf.c update_magnetic.c
        substep1_x.c substep1_y.c substep1_z.c substep2_a.c substep2_b.c
        substep3.c divrho.c stockholm.c vanleer_x_a.c vanleer_x_b.c
        vanleer_y_a.c vanleer_y_b.c vanleer_z_a.c vanleer_z_b.c
        momenta_x.c momenta_y.c momenta_z.c update_x.c update_y.c
        update_z.c update_density_x.c update_density_y.c update_density_z.c
        newvel_x.c newvel_y.c newvel_z.c advect_shift.c compute_residual.c
        change_frame.c potential.c correct_vtheta.c cfl.c compute_force.c
        lorentz_force.c CompPresPoly.c fargo_ppa_a.c fargo_ppa_b.c
        fargo_ppa_c.c fargo_ppa_d.c fargo_ppa_d_2d.c integrate_emf.c
        visctensor_cart.c addviscosity_cart.c addviscosity_cyl.c
        visctensor_cyl.c addviscosity_sph.c mon_dens.c mon_momx.c
        mon_momy.c mon_momz.c mon_torque.c checkmute_y.c checkmute_z.c
        mon_reynolds.c mon_maxwell.c visctensor_sph.c mon_bxflux.c
        fargo_ppa_steep.c setuphook1.c fill_ghostsx.c
        totaldens.c floor.c resetfield.c dust_diffusion_core.c
        dust_diffusion_coefficients.c nimhd_hall_effect_emfx.c
        nimhd_hall_effect_emfy.c nimhd_hall_effect_emfz.c
        nimhd_hall_effect_updateB.c nimhd_hall_effect_update_emfs.c
        nimhd_hall_effect_coeff.c nimhd_ambipolar_diffusion_emfx.c
        nimhd_ambipolar_diffusion_emfy.c nimhd_ambipolar_diffusion_emfz.c
        nimhd_ambipolar_diffusion_coeff.c nimhd_ohmic_diffusion.c
        nimhd_ohmic_diffusion_coeff.c compute_jx.c compute_jy.c compute_jz.c
        compute_vmed.c ram_advect.c ram_compute_ustar.c ram_plm.c
        ram_advect_lin.c)

    list(TRANSFORM gpu_sources_in REPLACE "\\.c" ".cu"
        OUTPUT_VARIABLE gpu_sources_out)
    list(TRANSFORM gpu_sources_in PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
    list(TRANSFORM gpu_sources_out PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)

    list(TRANSFORM boundary_outputs REPLACE "\\.c" ".cu"
        OUTPUT_VARIABLE gpu_boundary_outputs)

    # compile .c to .cu
    add_custom_command(OUTPUT ${gpu_sources_out} ${gpu_boundary_outputs}
        COMMAND ${Python_EXECUTABLE} ARGS ${script_dir}/c2cuda.py
        ${gpu_sources_in} ${boundary_outputs} -o ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${script_dir}/c2cuda.py ${gpu_sources_in} ${boundary_outputs}
        COMMENT "Compiling .c -> .cu")
    add_custom_target(generate_cuda DEPENDS
        ${gpu_sources_out} ${gpu_boundary_outputs})

    add_library(fargo3d_cuda ${gpu_sources_out} ${gpu_boundary_outputs}
        lowtasks.cu reduction_sum_device.cu reduction_min_device.cu)
    add_dependencies(fargo3d_cuda generate_cuda)
    target_include_directories(fargo3d_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    if(NFLUIDS GREATER 1)
        set(collision_kernel_template ${std_dir}/collisions_template.cu)
        set(collision_kernel ${CMAKE_CURRENT_BINARY_DIR}/collisions.cu)
        add_custom_command(OUTPUT ${collision_kernel} COMMAND ${Python_EXECUTABLE}
            ARGS ${script_dir}/collisions_gpu.py ${collision_kernel_template}
            -n ${NFLUIDS} -o ${collision_kernel}
            DEPENDS ${script_dir}/collisions_gpu.py ${collision_kernel_template})
        add_custom_target(generate_collision_kernel DEPENDS ${collision_kernel})
        add_dependencies(fargo3d_cuda generate_collision_kernel)
		target_sources(fargo3d_cuda PUBLIC ${collision_kernel})
    endif()

    target_link_libraries(fargo3d PUBLIC fargo3d_cuda)
    target_link_libraries(fargo3d PUBLIC CUDA::cudart)
    target_link_libraries(fargo3d_cuda PUBLIC CUDA::cudart)
    target_link_libraries(fargo3d_generated PUBLIC CUDA::cudart)
endif()
