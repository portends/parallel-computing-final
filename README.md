# parallel-computing-final

## To compile and run on OSC owen machine build and run inside the src folder
- module load cuda
- nvcc fire_sim.cu -lm -lGL -lGLU -lglut -o fire_sim
- ./fire_sim -g
- ./fire_sim -s
