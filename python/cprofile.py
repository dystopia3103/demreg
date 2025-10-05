import cProfile
 
def run_dem():
    return dn2dem_pos(dn_in2d,edn_in2d,trmatrix,tresp_logt,temps)
 
# Warm up
for _ in range(3): 
    _ = run_dem()
 
# Time it
%timeit run_dem()
 
# Run profiler
cProfile.run('run_dem()', 'profile_output_syn.prof')
# Now do the DEM maps calculation           
dem2d,edem2d,elogt2d,chisq2d,dn_reg2d=run_dem()