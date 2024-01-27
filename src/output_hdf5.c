#include <hdf5.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "fargo3d.h"

extern int Id_Var;
extern Param Var_Set[];

#ifdef FLOAT
/* single-precision mode */
#define REALTYPE        (H5T_NATIVE_FLOAT)
#else
/* double-precision mode */
#define REALTYPE        (H5T_NATIVE_DOUBLE)
#endif

#define FIELDS_PER_PLANET     (8)

static int fields_per_fluid = 2;
static int static_fields_per_fluid = 1;

static struct {
  MPI_Comm world_comm;

  hid_t file_id;          /* file handle */
  hid_t xfpl_id;          /* data transfer prop list */

  struct {
    hid_t group_id;
    hid_t memspace_id;
    hid_t **dataset_ids;  /* planet output dataset handles */
  } fluids;

#ifdef STOCKHOLM
  struct {
    hid_t group_id;       /* stockholm group handle */
    hid_t **dataset_ids;  /* stockholm dataset handles */
    hid_t memspace_id;
  } stockholm;            /* handles for stockholm boundary output */
#endif

  struct {
    hid_t dtype_id;       /* custom datatype for planets */
    hid_t memspace_id;    /* memory dataspace handle */
    hid_t *dataset_ids;   /* planet output dataset handles */
  } planets;              /* handles for planet output */

  hsize_t global_file_dims[4];
  hsize_t global_file_maxdims[4];
  hsize_t local_mem_dims[4];
  hsize_t write_dims[4];
  hsize_t ghost_dims[4];
  hsize_t chunk_dims[4];

  hsize_t global_file_start[4];
  hsize_t global_file_ghost_start[4];
} hdf5;

static int WriteFieldStatic(hid_t dset, void *buffer);
static int WriteFieldTimeDep(hid_t dset, void *buffer);

static int WriteRealAttribute(const char *name, real val);
static int WriteIntAttribute(const char *name, int val);
static int WriteBoolAttribute(const char *name, boolean val);
static int WriteStringAttribute(const char *name, const char *val);

int SetupOutputHdf5() {
  size_t len;
  char *fname;
  int rank = 0;

  htri_t avail;           /* flag to check for plugins */
  hid_t fapl_id;          /* file access prop list */

  hid_t fluids_dcpl_id;       /* dataset creation prop list */
  hid_t fluids_group_id;
  hid_t fluids_filespace_id;
  hid_t *fluids_subgroup_ids; /* group handles for fluid fields */
#ifdef STOCKHOLM
  hid_t stockholm_dcpl_id;
  hid_t stockholm_filespace_id;
  hid_t *stockholm_subgroup_ids; /* group handles for stockholm fields */
#endif

#ifdef Z
  fields_per_fluid += 1;
  static_fields_per_fluid += 1;
#endif
#ifdef Y
  fields_per_fluid += 1;
  static_fields_per_fluid += 1;
#endif
#ifdef X
  fields_per_fluid += 1;
  static_fields_per_fluid += 1;
#endif
#ifdef ADIABATIC
  static_fields_per_fluid += 1;
#endif

  MPI_Comm_dup(MPI_COMM_WORLD, &(hdf5.world_comm));

  /* form the output filename from the directory and a user-supplied tag */
  len = strlen(OUTPUTDIR) + strlen(FILETAG) + 5;
  fname = malloc(len * sizeof(char));
  snprintf(fname, len, "%s/%s.h5", OUTPUTDIR, FILETAG);

  fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  hdf5.xfpl_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef PARALLEL
  /* if we're working in parallel, we need to tell the library; this allows
   * writing to the same file from multiple processes */
  H5Pset_fapl_mpio(fapl_id, hdf5.world_comm, MPI_INFO_NULL);
  H5Pset_dxpl_mpio(hdf5.xfpl_id, H5FD_MPIO_COLLECTIVE);
#endif
  H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

  /* create the file handle */
  hdf5.file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  free(fname);
  if (hdf5.file_id < 0) return -1;

  /* set up for fluid output */
  fluids_group_id = H5Gcreate(hdf5.file_id,
      "fluids", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (hdf5.fluids.group_id < 0) return -1;

  /* 0. add a time dimension */
  hdf5.global_file_dims[rank] = 0;
  hdf5.global_file_maxdims[rank] = H5S_UNLIMITED;
  hdf5.chunk_dims[rank] = 1;
  hdf5.local_mem_dims[rank] = 1;
  hdf5.write_dims[rank] = 1;
  hdf5.ghost_dims[rank] = 0;
  rank += 1;

  /* 1. if enabled, add z dimension */
#ifdef Z
  hdf5.global_file_dims[rank] = NZ;
  hdf5.write_dims[rank] = Nz;
  hdf5.ghost_dims[rank] = Nz + NGHZ * (Gridd.K == 0) +
    NGHZ * (Gridd.K + 1 == Gridd.NK);
#ifdef WRITEGHOSTS
  hdf5.global_file_dims[rank] += 2 * NGHZ;
  hdf5.write_dims[rank] += NGHZ * (Gridd.K == 0) +
    NGHZ * (Gridd.K + 1 == Gridd.NK);
#endif
  hdf5.global_file_maxdims[rank] =
    hdf5.chunk_dims[rank] = hdf5.global_file_dims[rank];
  hdf5.local_mem_dims[rank] = Nz + 2 * NGHZ;
  rank += 1;
#endif  // Z

    /* 2. if enabled, add y dimension */
#ifdef Y
  hdf5.global_file_dims[rank] = NY;
  hdf5.write_dims[rank] = Ny;
  hdf5.ghost_dims[rank] = Ny + NGHY * (Gridd.J == 0) +
    NGHY * (Gridd.J + 1 == Gridd.NJ);
#ifdef WRITEGHOSTS
  hdf5.global_file_dims[rank] += 2 * NGHY;
  hdf5.write_dims[rank] += NGHY * (Gridd.J == 0) +
    NGHY * (Gridd.J + 1 == Gridd.NJ);
#endif
  hdf5.global_file_maxdims[rank] =
    hdf5.chunk_dims[rank] = hdf5.global_file_dims[rank];
  hdf5.local_mem_dims[rank] = Ny + 2 * NGHY;
  rank += 1;
#endif  // Y

  /* 3. if enabled, add x dimension; note that the x dimension isn't
   * split over multiple ranks */
#ifdef X
  hdf5.global_file_dims[rank] = NX;
  hdf5.write_dims[rank] = Nx;
  hdf5.ghost_dims[rank] = Nx + 2 * NGHX;
#ifdef WRITEGHOSTS
  hdf5.global_file_dims[rank] += 2 * NGHX;
  hdf5.write_dims[rank] += 2 * NGHX;
#endif
  hdf5.global_file_maxdims[rank] =
    hdf5.chunk_dims[rank] = hdf5.global_file_dims[rank];
  hdf5.local_mem_dims[rank] = Nx + 2 * NGHX;
  rank += 1;
#endif  // X

  /* filespace expands with time, memspace does not */
  fluids_filespace_id = H5Screate_simple(rank,
    hdf5.global_file_dims, hdf5.global_file_maxdims);
  hdf5.fluids.memspace_id = H5Screate_simple(rank, hdf5.local_mem_dims, NULL);

#ifdef STOCKHOLM
  /* stockholm output isn't time-dependent */
  stockholm_filespace_id = H5Screate_simple(rank - 2,
    hdf5.global_file_dims + 1, hdf5.global_file_maxdims + 1);
  hdf5.stockholm.memspace_id = H5Screate_simple(rank - 2,
    hdf5.local_mem_dims + 1, NULL);
#endif

  /* unlimited-dimension datasets **must** be chunked */
  fluids_dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(fluids_dcpl_id, rank, hdf5.chunk_dims);
  avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (avail && (COMPRESSLEVEL > 0)) {
    /* conditionally enable compression */
    H5Pset_deflate(fluids_dcpl_id, COMPRESSLEVEL);
  }

#ifdef STOCKHOLM
  stockholm_dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(stockholm_dcpl_id, rank - 2, hdf5.chunk_dims + 1);
  avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (avail && (COMPRESSLEVEL > 0)) {
    /* conditionally enable compression */
    H5Pset_deflate(stockholm_dcpl_id, COMPRESSLEVEL);
  }
#endif

  /* allocate memory to store group handles */
  fluids_subgroup_ids = malloc(NFLUIDS * sizeof(hid_t));
  if (fluids_subgroup_ids == NULL) return -1;
#ifdef STOCKHOLM
  stockholm_subgroup_ids = malloc(NFLUIDS * sizeof(hid_t));
  if (stockholm_subgroup_ids == NULL) return -1;
#endif

  /* allocate memory to store dataset handles */
  hdf5.fluids.dataset_ids = malloc(NFLUIDS * sizeof(hid_t *));
  if (hdf5.fluids.dataset_ids == NULL) return -1;
#ifdef STOCKHOLM
  hdf5.stockholm.dataset_ids = malloc(NFLUIDS * sizeof(hid_t *));
  if (hdf5.stockholm.dataset_ids == NULL) return -1;
#endif

  for (int f = 0; f < NFLUIDS; ++f) {
    int d;        /* local counter for datasets */
    hid_t gid;    /* local group handle */
    hid_t subgid; /* local subgroup handle */

    /* create a group for each fluid */
    gid = H5Gcreate(fluids_group_id, Fluids[f]->name,
      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (gid < 0) return -1;

    /* allocate memory for dataset handles */
    hdf5.fluids.dataset_ids[f] = malloc(fields_per_fluid * sizeof(hid_t));

    d = 0; /* create a dataset for each field */
    hdf5.fluids.dataset_ids[f][d++] = H5Dcreate(gid, "dens", REALTYPE,
      fluids_filespace_id, H5P_DEFAULT, fluids_dcpl_id, H5P_DEFAULT);
#ifdef Z
    hdf5.fluids.dataset_ids[f][d++] = H5Dcreate(gid, "zvel", REALTYPE,
      fluids_filespace_id, H5P_DEFAULT, fluids_dcpl_id, H5P_DEFAULT);
#endif
#ifdef Y
    hdf5.fluids.dataset_ids[f][d++] = H5Dcreate(gid, "yvel", REALTYPE,
      fluids_filespace_id, H5P_DEFAULT, fluids_dcpl_id, H5P_DEFAULT);
#endif
#ifdef X
    hdf5.fluids.dataset_ids[f][d++] = H5Dcreate(gid, "xvel", REALTYPE,
      fluids_filespace_id, H5P_DEFAULT, fluids_dcpl_id, H5P_DEFAULT);
#endif
    hdf5.fluids.dataset_ids[f][d++] = H5Dcreate(gid, "ener", REALTYPE,
      fluids_filespace_id, H5P_DEFAULT, fluids_dcpl_id, H5P_DEFAULT);

#ifdef STOCKHOLM
    d = 0; /* stockholm group within each fluid */
    subgid = H5Gcreate(gid, "stockholm", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (subgid < 0) return -1;

    /* allocate memory for dataset handles */
    hdf5.stockholm.dataset_ids[f] = malloc(static_fields_per_fluid * sizeof(hid_t));

    hdf5.stockholm.dataset_ids[f][d++] = H5Dcreate(subgid, "dens", REALTYPE,
      stockholm_filespace_id, H5P_DEFAULT, stockholm_dcpl_id, H5P_DEFAULT);
#ifdef Z
    hdf5.stockholm.dataset_ids[f][d++] = H5Dcreate(subgid, "zvel", REALTYPE,
      stockholm_filespace_id, H5P_DEFAULT, stockholm_dcpl_id, H5P_DEFAULT);
#endif
#ifdef Y
    hdf5.stockholm.dataset_ids[f][d++] = H5Dcreate(subgid, "yvel", REALTYPE,
      stockholm_filespace_id, H5P_DEFAULT, stockholm_dcpl_id, H5P_DEFAULT);
#endif
#ifdef X
    hdf5.stockholm.dataset_ids[f][d++] = H5Dcreate(subgid, "xvel", REALTYPE,
      stockholm_filespace_id, H5P_DEFAULT, stockholm_dcpl_id, H5P_DEFAULT);
#endif
#ifdef ADIABATIC
    hdf5.stockholm.dataset_ids[f][d++] = H5Dcreate(subgid, "ener", REALTYPE,
      stockholm_filespace_id, H5P_DEFAULT, stockholm_dcpl_id, H5P_DEFAULT);
#endif

    H5Gclose(subgid);     /* close temporary handle */
#endif  // STOCKHOLM

    H5Gclose(gid);        /* close temporary handle */
  }

    /* can close these handles, datasets are created */
  H5Pclose(fapl_id);
  H5Pclose(fluids_dcpl_id);
  H5Gclose(fluids_group_id);
  H5Sclose(fluids_filespace_id);
#ifdef STOCKHOLM
  H5Pclose(stockholm_dcpl_id);
  H5Sclose(stockholm_filespace_id);
#endif

  if (ThereArePlanets && (Sys != NULL)) {
    size_t sz, count = 0;
    hid_t planets_filespace_id, group_id, dcpl_id;
    hsize_t chunkdim = 1, dim = 0, maxdim = H5S_UNLIMITED;

    group_id = H5Gcreate(hdf5.file_id, "planets",
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) return -1;

    dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl_id, 1, &chunkdim);
    avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
    if (avail && (COMPRESSLEVEL > 0)) {
        H5Pset_deflate(dcpl_id, COMPRESSLEVEL);
    }

    /* planet data is essentially a big array of "scalars" (actually
     * compound structs), so only need rank 1 */
    planets_filespace_id = H5Screate_simple(1, &dim, &maxdim);
    hdf5.planets.memspace_id = H5Screate(H5S_SCALAR);

    sz = H5Tget_size(REALTYPE);
    hdf5.planets.dtype_id = H5Tcreate(H5T_COMPOUND, FIELDS_PER_PLANET * sz);

    H5Tinsert(hdf5.planets.dtype_id, "t",    sz * (count++), REALTYPE);
    H5Tinsert(hdf5.planets.dtype_id, "mass", sz * (count++), REALTYPE);

    H5Tinsert(hdf5.planets.dtype_id, "x", sz * (count++), REALTYPE);
    H5Tinsert(hdf5.planets.dtype_id, "y", sz * (count++), REALTYPE);
    H5Tinsert(hdf5.planets.dtype_id, "z", sz * (count++), REALTYPE);

    H5Tinsert(hdf5.planets.dtype_id, "xvel", sz * (count++), REALTYPE);
    H5Tinsert(hdf5.planets.dtype_id, "yvel", sz * (count++), REALTYPE);
    H5Tinsert(hdf5.planets.dtype_id, "zvel", sz * (count++), REALTYPE);

    /* create a dataset per planet in the system */
    hdf5.planets.dataset_ids = malloc(Sys->nb * sizeof(hid_t));
    if (hdf5.planets.dataset_ids == NULL) return -1;

    for (int pl = 0; pl < Sys->nb; ++pl) {
      hdf5.planets.dataset_ids[pl] = H5Dcreate(group_id, Sys->name[pl],
        hdf5.planets.dtype_id, planets_filespace_id, H5P_DEFAULT,
        dcpl_id, H5P_DEFAULT);
    }

    /* close temporary handles */
    H5Pclose(dcpl_id);
    H5Gclose(group_id);
    H5Sclose(planets_filespace_id);
  }

  {
    /* a little trick for determining which indices should be written
     * by each processor, using the grid indices that have already been
     * assigned and an exclusive sum */
    int rank = 0;
    MPI_Comm tmp_comm;

    /* time dimension */
    hdf5.global_file_start[rank] = 0;
    hdf5.global_file_ghost_start[rank] = 0;
    rank += 1;

#ifdef Z
    MPI_Comm_split(hdf5.world_comm, Gridd.J, CPU_Rank, &tmp_comm);
    MPI_Exscan(&(hdf5.write_dims[rank]), &(hdf5.global_file_start[rank]),
      1, MPI_UNSIGNED_LONG, MPI_SUM, tmp_comm);
    MPI_Exscan(&(hdf5.ghost_dims[rank]), &(hdf5.global_file_ghost_start[rank]),
      1, MPI_UNSIGNED_LONG, MPI_SUM, tmp_comm);
    MPI_Comm_free(&tmp_comm);
    rank += 1;
#endif

#ifdef Y
    MPI_Comm_split(hdf5.world_comm, Gridd.K, CPU_Rank, &tmp_comm);
    MPI_Exscan(&(hdf5.write_dims[rank]), &(hdf5.global_file_start[rank]),
      1, MPI_UNSIGNED_LONG, MPI_SUM, tmp_comm);
    MPI_Exscan(&(hdf5.ghost_dims[rank]), &(hdf5.global_file_ghost_start[rank]),
      1, MPI_UNSIGNED_LONG, MPI_SUM, tmp_comm);
    MPI_Comm_free(&tmp_comm);
    rank += 1;
#endif

#ifdef X
    hdf5.global_file_start[rank] = 0;
    hdf5.global_file_ghost_start[rank] = 0;
    rank += 1;
#endif
  }

  return 0;
}

int WriteDomainHdf5() {
  hid_t filespace_id, memspace_id;
  hid_t group_id, dataset_id, dcpl_id;
  hsize_t memsz, memst, memct, filesz;

  int rank = 1;

  MPI_Barrier(hdf5.world_comm);

  group_id = H5Gcreate(hdf5.file_id,
    "domain", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (group_id < 0) return -1;

#ifdef Z
  /* memory start index, memory count, memory total size, and file total size
   * for the z-coordinate data */
  memst = NGHZ * (Gridd.K > 0);
  memct = Nz + NGHZ * (Gridd.K == 0) + (NGHZ + 1) * (Gridd.K + 1 == Gridd.NK);
  memsz = Nz + 2 * NGHZ + 1;
  filesz = NZ + 2 * NGHZ + 1;

  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 1, &memsz);

  filespace_id = H5Screate_simple(1, &filesz, NULL);
  memspace_id = H5Screate_simple(1, &memsz, NULL);

  dataset_id = H5Dcreate(group_id, "z", REALTYPE,
    filespace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  if (Gridd.J == 0) {
    /* only one rank should write this data, otherwise it gets re-written
     * for every processor in y */
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
      &(hdf5.global_file_ghost_start[rank]), NULL, &memct, NULL);
    H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
      &memst, NULL, &memct, NULL);
  } else {
    /* to keep all other processors from writing, since H5Dwrite must be
     * collective, just clear the hyperslab selection */
    H5Sselect_none(filespace);
    H5Sselect_none(memspace);
  }

  /* do the write operation */
  H5Dwrite(dataset_id, REALTYPE, memspace_id, filespace_id, hdf5.xfpl_id, Zmin);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);

  rank += 1;
#endif
#ifdef Y
  /* memory start index, memory count, memory total size, and file total size
   * for the y-coordinate data */
  memst = NGHY * (Gridd.J > 0);
  memct = Ny + NGHY * (Gridd.J == 0) + (NGHY + 1) * (Gridd.J + 1 == Gridd.NJ);
  memsz = Ny + 2 * NGHY + 1;
  filesz = NY + 2 * NGHY + 1;

  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 1, &memsz);

  filespace_id = H5Screate_simple(1, &filesz, NULL);
  memspace_id = H5Screate_simple(1, &memsz, NULL);

  dataset_id = H5Dcreate(group_id, "y", REALTYPE,
    filespace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  if (Gridd.K == 0) {
    /* only one rank should write this data, otherwise it gets re-written
     * for every processor in y */
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
      &(hdf5.global_file_ghost_start[rank]), NULL, &memct, NULL);
    H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
      &memst, NULL, &memct, NULL);
  } else {
    /* to keep all other processors from writing, since H5Dwrite must be
     * collective, just clear the hyperslab selection */
    H5Sselect_none(filespace_id);
    H5Sselect_none(memspace_id);
  }

  /* do the write operation */
  H5Dwrite(dataset_id, REALTYPE, memspace_id, filespace_id, hdf5.xfpl_id, Ymin);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);

  rank += 1;
#endif
#ifdef X
  /* memory start index and memory total size for the x-coordinate data */
  memst = 0;
  memsz = Nx + 2 * NGHX + 1;

  dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 1, &memsz);

  filespace_id = H5Screate_simple(1, &memsz, NULL);
  memspace_id = H5Screate_simple(1, &memsz, NULL);

  dataset_id = H5Dcreate(group_id, "x", REALTYPE,
    filespace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  if ((Gridd.J == 0) && (Gridd.K == 0)) {
    /* the x-coordinate isn't split across ranks; limit writing to just
     * the (0,0) rank, which is always guaranteed to exist, even for
     * sequential runs */
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
      &memst, NULL, &memsz, NULL);
    H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET,
      &memst, NULL, &memsz, NULL);
  } else {
    /* to keep all other processors from writing, since H5Dwrite must be
     * collective, just clear the hyperslab selection */
    H5Sselect_none(filespace_id);
    H5Sselect_none(memspace_id);
  }

  H5Dwrite(dataset_id, REALTYPE, memspace_id, filespace_id, hdf5.xfpl_id, Xmin);

  H5Pclose(dcpl_id);
  H5Dclose(dataset_id);
  H5Sclose(memspace_id);
  H5Sclose(filespace_id);

  rank += 1;
#endif

  /* close the group, we're done with it */
  H5Gclose(group_id);
  /* make sure all ranks are done writing */
  MPI_Barrier(hdf5.world_comm);

  return 0;
}

int WriteFieldTimeDep(hid_t dset_id, void *buffer) {
  /* Write a time-dependent field (maximum rank 4, time + z + y + x)
   * to the open file in a multiprocess-safe way. */
  int err, rank = 0;
  hid_t filespace_id;
  hsize_t file_dims[4], file_start[4];

  /* get the "old" dataspace global dimensions */
  filespace_id = H5Dget_space(dset_id);
  H5Sget_simple_extent_dims(filespace_id, file_dims, NULL);
  H5Sclose(filespace_id);

  /* increase the first dimension by 1 for this timestep */
  file_start[rank++] = file_dims[0];
  file_dims[0] += 1;
  H5Dset_extent(dset_id, file_dims);
  filespace_id = H5Dget_space(dset_id);

  /* other start indices are from precomputed arrays */
#ifdef Z
  file_start[rank++] = hdf5.global_file_start[rank];
#endif
#ifdef Y
  file_start[rank++] = hdf5.global_file_start[rank];
#endif
#ifdef X
  file_start[rank++] = 0;
#endif

  /* select this processor's slab and write the data */
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
    file_start, NULL, hdf5.write_dims, NULL);
  err = H5Dwrite(dset_id, REALTYPE, hdf5.fluids.memspace_id,
    filespace_id, hdf5.xfpl_id, buffer);
  H5Sclose(filespace_id);

  return err;
}

int WriteFieldStatic(hid_t dset_id, void *buffer) {
  /* Write a time-independent field (maximum rank 2, z + y)
   * to the open file in a multiprocess-safe way. This is
   * only used for the Stockholm boundary data. */
  int err = 0;
#ifdef STOCKHOLM
  int rank = 1;
  hid_t filespace_id;
  hsize_t file_dims[4], file_start[4];

  /* get the dataspace but don't modify */
  filespace_id = H5Dget_space(dset_id);

#ifdef Z
  file_start[rank++] = hdf5.global_file_start[rank];
#endif
#ifdef Y
  file_start[rank++] = hdf5.global_file_start[rank];
#endif

  /* select this processor's slab and write the data */
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
    file_start + 1, NULL, hdf5.write_dims + 1, NULL);
  err = H5Dwrite(dset_id, REALTYPE, hdf5.stockholm.memspace_id,
    filespace_id, hdf5.xfpl_id, buffer);
  H5Sclose(filespace_id);
#endif
  return err;
}

int WriteOutputsHdf5() {
  int err, rank = 0;
  hsize_t mem_start[4];

  mem_start[rank++] = 0;
#ifdef Z
#ifdef WRITEGHOSTS
  mem_start[rank++] = NGHZ * (Gridd.K > 0);
#else
  mem_start[rank++] = NGHZ;
#endif  // WRITEGHOSTS
#endif  // Z
#ifdef Y
#ifdef WRITEGHOSTS
  mem_start[rank++] = NGHY * (Gridd.J > 0);
#else
  mem_start[rank++] = NGHY;
#endif  // WRITEGHOSTS
#endif  // Y
#ifdef X
#ifdef WRITEGHOSTS
  mem_start[rank++] = 0;
#else
  mem_start[rank++] = NGHX;
#endif  // WRITEGHOSTS
#endif  // X

  MPI_Barrier(hdf5.world_comm);
  H5Sselect_hyperslab(hdf5.fluids.memspace_id, H5S_SELECT_SET,
    mem_start, NULL, hdf5.write_dims, NULL);

  for (int f = 0; f < NFLUIDS; ++f) {
    int d = 0;  /* local dataset counter */
    hid_t *dset_ids = hdf5.fluids.dataset_ids[f];

    err = WriteFieldTimeDep(dset_ids[d++], Fluids[f]->Density->field_cpu);
    if (err < 0) return err;
#ifdef Z
    err = WriteFieldTimeDep(dset_ids[d++], Fluids[f]->Vz->field_cpu);
    if (err < 0) return err;
#endif
#ifdef Y
    err = WriteFieldTimeDep(dset_ids[d++], Fluids[f]->Vy->field_cpu);
    if (err < 0) return err;
#endif
#ifdef X
    err = WriteFieldTimeDep(dset_ids[d++], Fluids[f]->Vx->field_cpu);
    if (err < 0) return err;
#endif
    err = WriteFieldTimeDep(dset_ids[d++], Fluids[f]->Energy->field_cpu);
    if (err < 0) return err;
  }

  /* wait until all ranks finish, then flush to disk */
  MPI_Barrier(hdf5.world_comm);
  H5Fflush(hdf5.file_id, H5F_SCOPE_GLOBAL);

  return 0;
}

int WriteOutputs2dHdf5() {
#ifdef STOCKHOLM
  int err, rank = 1;
  hsize_t mem_start[4];

  MPI_Barrier(hdf5.world_comm);

#ifdef Z
#ifdef WRITEGHOSTS
  mem_start[rank++] = NGHZ * (Gridd.K == 0);
#else
  mem_start[rank++] = NGHZ;
#endif  // WRITEGHOSTS
#endif  // Z
#ifdef Y
#ifdef WRITEGHOSTS
  mem_start[rank++] = NGHY * (Gridd.J == 0);
#else
  mem_start[rank++] = NGHY;
#endif  // WRITEGHOSTS
#endif  // Y

  H5Sselect_hyperslab(hdf5.stockholm.memspace_id, H5S_SELECT_SET,
    mem_start + 1, NULL, hdf5.write_dims + 1, NULL);

  for (int f = 0; f < NFLUIDS; ++f) {
    int d = 0;  /* local dataset counter */
    hid_t *dset_ids = hdf5.stockholm.dataset_ids[f];

    err = WriteFieldStatic(dset_ids[d++], Fluids[f]->Density0->field_cpu);
    if (err < 0) return err;
#ifdef Z
    err = WriteFieldStatic(dset_ids[d++], Fluids[f]->Vz0->field_cpu);
    if (err < 0) return err;
#endif
#ifdef Y
    err = WriteFieldStatic(dset_ids[d++], Fluids[f]->Vy0->field_cpu);
    if (err < 0) return err;
#endif
#ifdef X
    err = WriteFieldStatic(dset_ids[d++], Fluids[f]->Vx0->field_cpu);
    if (err < 0) return err;
#endif
#ifdef ADIABATIC
    err = WriteFieldStatic(dset_ids[d++], Fluids[f]->Energy0->field_cpu);
    if (err < 0) return err;
#endif
  }

  /* wait until all ranks finish, then flush to disk */
  MPI_Barrier(hdf5.world_comm);
  H5Fflush(hdf5.file_id, H5F_SCOPE_GLOBAL);
#endif

  return 0;
}

int WritePlanetsHdf5() {
  int err;
  hid_t filespace_id;
  hsize_t file_dims, file_start, file_count = 1;
  real planet_buffer[FIELDS_PER_PLANET];

  for (int pl = 0; pl < Sys->nb; ++pl) {
    int i = 0;  /* local field counter */
    hid_t dset_id = hdf5.planets.dataset_ids[pl];

    /* get the "old" dataspace global dimensions */
    filespace_id = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(filespace_id, &file_dims, NULL);
    H5Sclose(filespace_id);

    /* increase the first (and only) dimension by 1 for this timestep */
    file_start = file_dims;
    file_dims += 1;
    H5Dset_extent(dset_id, &file_dims);
    filespace_id = H5Dget_space(dset_id);

    planet_buffer[i++] = PhysicalTime;
    planet_buffer[i++] = Sys->mass[pl];

    planet_buffer[i++] = Sys->x[pl];
    planet_buffer[i++] = Sys->y[pl];
    planet_buffer[i++] = Sys->z[pl];

    planet_buffer[i++] = Sys->vx[pl];
    planet_buffer[i++] = Sys->vy[pl];
    planet_buffer[i++] = Sys->vz[pl];

    if (CPU_Master) {
      /* only one rank should write the data */
      H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET,
        &file_start, NULL, &file_count, NULL);
    } else {
      /* others call H5Dwrite with no elements selected */
      H5Sselect_none(filespace_id);
      H5Sselect_none(hdf5.planets.memspace_id);
    }

    /* write the data */
    err = H5Dwrite(dset_id, hdf5.planets.dtype_id, hdf5.planets.memspace_id,
      filespace_id, hdf5.xfpl_id, planet_buffer);
    if (err < 0) return err;

    H5Sclose(filespace_id);
  }

  return 0;
}

int WriteRealAttribute(const char *name, real val) {
  herr_t err;
  hid_t att_id, space_id;

  space_id = H5Screate(H5S_SCALAR);
  if (space_id < 0) return -1;

  att_id = H5Acreate(hdf5.file_id, name,
    REALTYPE, space_id, H5P_DEFAULT, H5P_DEFAULT);
  if (att_id < 0) return -1;

  err = H5Awrite(att_id, REALTYPE, &val);
  if (err < 0) return -1;

  H5Aclose(att_id);
  H5Sclose(space_id);

  return 0;
}

int WriteIntAttribute(const char *name, int val) {
  herr_t err;
  hid_t att_id, space_id;

  space_id = H5Screate(H5S_SCALAR);
  if (space_id < 0) return -1;

  att_id = H5Acreate(hdf5.file_id, name,
    H5T_STD_I32BE, space_id, H5P_DEFAULT, H5P_DEFAULT);
  if (att_id < 0) return -1;

  err = H5Awrite(att_id, H5T_NATIVE_INT, &val);
  if (err < 0) return -1;

  H5Aclose(att_id);
  H5Sclose(space_id);

  return 0;
}

int WriteBoolAttribute(const char *name, boolean val) {
  herr_t err;
  hid_t att_id, space_id;

  space_id = H5Screate(H5S_SCALAR);
  if (space_id < 0) return -1;

  att_id = H5Acreate(hdf5.file_id, name,
    H5T_STD_B8BE, space_id, H5P_DEFAULT, H5P_DEFAULT);
  if (att_id < 0) return -1;

  err = H5Awrite(att_id, H5T_NATIVE_B8, &val);
  if (err < 0) return -1;

  H5Aclose(att_id);
  H5Sclose(space_id);

  return 0;
}

int WriteStringAttribute(const char *name, const char *val) {
  herr_t err;
  hid_t att_id, space_id, type_id;
  size_t len;

  space_id = H5Screate(H5S_SCALAR);
  if (space_id < 0) return -1;

  type_id = H5Tcopy(H5T_C_S1);
  if (type_id < 0) return -1;

  len = strlen(val);
  err = H5Tset_size(type_id, len);
  if (err < 0) return -1;

  att_id = H5Acreate(hdf5.file_id, name,
    type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
  if (att_id < 0) return -1;

  err = H5Awrite(att_id, type_id, val);
  if (err < 0) return -1;

  H5Aclose(att_id);
  H5Tclose(type_id);
  H5Sclose(space_id);

  return 0;
}

int WriteParametersHdf5() {
  for (int i = 0; i < Id_Var; i++) {
    char *var = Var_Set[i].variable;
    const char *name = Var_Set[i].name;

    switch (Var_Set[i].type) {
      case REAL:
        WriteRealAttribute(name, *((real *) var));
        break;

      case INT:
        WriteIntAttribute(name, *((int *) var));
        break;

      case BOOL:
        WriteBoolAttribute(name, *((boolean *) var));
        break;

      case STRING:
        WriteStringAttribute(name, var);
        break;
    }
  }

  return 0;
}

void TeardownOutputHdf5() {
  H5Pclose(hdf5.xfpl_id);
  H5Sclose(hdf5.fluids.memspace_id);
#ifdef STOCKHOLM
  H5Sclose(hdf5.stockholm.memspace_id);
#endif

  for (int f = 0; f < NFLUIDS; ++f) {
    for (int i = 0; i < fields_per_fluid; ++i) {
      H5Dclose(hdf5.fluids.dataset_ids[f][i]);
    }

    free(hdf5.fluids.dataset_ids[f]);

#ifdef STOCKHOLM
    for (int i = 0; i < static_fields_per_fluid; ++i) {
      H5Dclose(hdf5.stockholm.dataset_ids[f][i]);
    }

    free(hdf5.stockholm.dataset_ids[f]);
#endif
  }

  if (ThereArePlanets && (Sys != NULL)) {
    H5Tclose(hdf5.planets.dtype_id);
    H5Sclose(hdf5.planets.memspace_id);

    for (int pl = 0; pl < Sys->nb; ++pl) {
      H5Dclose(hdf5.planets.dataset_ids[pl]);
    }

    free(hdf5.planets.dataset_ids);
  }

  H5Fclose(hdf5.file_id);
  MPI_Comm_free(&(hdf5.world_comm));
}
