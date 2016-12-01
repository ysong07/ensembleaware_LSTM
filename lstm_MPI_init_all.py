import subprocess

subprocess.Popen(['mpiexec', '-n' ,'4', 'python', 'lstm_MPI_init.py'])
subprocess.Popen(['mpiexec', '-n' ,'4', 'python', 'lstm_MPI_init1.py'])

