#BSUB -q gpuqueue
#BSUB -J trial
#BSUB -m "ls-gpu lt-gpu"
#BSUB -n 4 -R "rusage[mem=4] span[ptile=4]"
#BSUB -gpu "num=4:j_exclusive=yes:mode=shared"
#BSUB -W 8:00
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

chmod +x explicit.yaml
# mpirun yank script --yaml=sams.yaml
build_mpirun_configfile "yank script --yaml=explicit.yaml"
mpiexec.hydra -f hostfile -configfile configfile
