from subprocess import run

#prog = "python3"
prog= '/Users/djin1/opt/miniconda3/bin/python3'

file_name = "Ch6-3a_Hint_of_multithreading_py3.py"

exec0 = "{} {}".format(prog, file_name)
print("> "+exec0)
run(exec0, shell=True)

exec1 = exec0+" 4"
print("\n> "+exec1)
run(exec1, shell=True)

options = [(4,1), (3,4), (2,4)]
for opt in options:
    exec2 = exec0+" {} {}".format(*opt)
    print("\n> "+exec2)
    run(exec2, shell=True)
    #run(exec2.split())
