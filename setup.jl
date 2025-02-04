
using Preferences, UUIDs

set_preferences!(
           UUID("7d669430-f675-4ae7-b43e-fab78ec5a902"), # UUID of P4est.jl
           "libp4est" => "/home/gridsan/mleprovost/julia/p4est_build/lib/libp4est.so", 
           force = true)

using P4est

P4est.set_library_sc!("/home/gridsan/mleprovost/julia/p4est_build/lib/libsc.so")

using MPIPreferences

MPIPreferences.use_system_binary()

using P4est, MPI; MPI.Init()
connectivity = p4est_connectivity_new_periodic()
p4est = p4est_new_ext(MPI.COMM_WORLD, connectivity, 0, 2, 0, 0, C_NULL, C_NULL)
p4est_obj = unsafe_load(p4est)
MPI.Barrier(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
@info "Setup" rank p4est_obj.local_num_quadrants p4est_obj.global_num_quadrants
p4est_destroy(p4est)
p4est_connectivity_destroy(connectivity)



using P4test, MPI

P4est.set_library_p4est!("/home/gridsan/mleprovost/julia/p4est_build/lib/libp4est.so")
P4est.set_library_sc!("/home/gridsan/mleprovost/julia/p4est_build/lib/libsc.so")


using MPIPreferences


using P4est, MPI; MPI.Init()



using T8code, MPI; MPI.Init()

T8code.set_library_t8code!("/home/gridsan/mleprovost/julia/t8code_build/lib/libt8.so")

T8code.set_library_p4est!("/home/gridsan/mleprovost/julia/t8code_build/lib/libp4est.so")

T8code.set_library_sc!("/home/gridsan/mleprovost/julia/t8code_build/lib/libsc.so")

using MPIPreferences

MPIPreferences.use_system_binary()

using T8code, MPI

using T8code.Libt8: sc_init
using T8code.Libt8: sc_finalize
using T8code.Libt8: SC_LP_ESSENTIAL
using T8code.Libt8: SC_LP_PRODUCTION

# Initialize MPI. This has to happen before we initialize sc or t8code.
mpiret = MPI.Init()

comm = MPI.COMM_WORLD

# Initialize the sc library, has to happen before we initialize t8code.
sc_init(comm, 1, 1, C_NULL, SC_LP_ESSENTIAL)

# Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels.
t8_init(SC_LP_PRODUCTION)




