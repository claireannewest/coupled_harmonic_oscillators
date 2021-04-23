# Confocal Photothermal Image Pipeline
This set of files creates confocal photothermal images by calculating the light scattered by metallic nanoparticle assemblies.

### Step 1. Create a shape file that contains a discretized version of the nanoparticle assembly. 
A sample fortran shapemaker file is included, called "shape.f90". If you choose not to use this file, make sure you update the scripts which submit the job to reflect the rastery, rasterz parameters. Otherwise,
* Adjust "shape.f90" to contain the shape that you would like to run. Make sure you do not change the variables "rastery" and "rasterz". These variables will also still need to be added onto your y and z directions. Note that to make the photothermal image, the shape is rastered across the beam centered at (0,0). 
### Step 2. Update the values in "parameters.input" to model the system of interest.
At a minimum, make sure you adjust: 
* Line 5 to the wavlength (in microns) of your desired pump / heating beam.
* Line 17 to the dipole spacing you've decided upon according to your shape.f90 file

### Step 3. Run a single test point to ensure no errors.
* This will launch a single calculation (i.e. a single raster position.)
* Adjust "ystart" and "z" in the file "launch_temp" to be the position (in lattice units) where you'd like your shape to be.
* To launch the calculation, simply type "sbatch launch_single.slurm" in the command line.

### Step 4. Identify the 2D image window.
* Change files "launch_temp1", "launch_temp2", and launch_ful.sh" accordingly.
* The variables "yrange", "ystart" in "launch_temp1" and "launch_temp2" should be updated to cover the y ranges you wish to span. 
* The variables "zrange", "zstart" in launch_full.sh should be updated to cover the z ranges.
* The variables "ss" in all three files should be identical. This is the step size and can be adjusted to take more / less points in the psf.

### Step 5. Luanch the simulations.
To launch, type "bash launch_full.sh" into command line.
    
   
