import re
import sys

ifile = open("../std/collisions_template.cu","r")
lines = ifile.readlines()
ifile.close()

ofile = open("collisions.cu","w")

nfluids = int(sys.argv[1])

for line in lines:
    if re.search("%FLUIDS0",line):
        newline = ""
        for i in range(nfluids):
            newline += "real* rho"+str(i)+",\n"
            newline += "real* v_input"+str(i)+",\n"
            if i == nfluids-1:
                newline += "real* v_output"+str(i)+"){\n"
            else:
                newline += "real* v_output"+str(i)+",\n"
        ofile.write(newline)
        continue
    
    if re.search("%FLUIDS1",line):
        newline = "real *rho[NFLUIDS] = {"
        for i in range(nfluids):
            if i == nfluids-1:
                newline += "rho"+str(i)
            else:
                newline += "rho"+str(i)+","
        newline += "};\n"
        ofile.write(newline)
        newline = "real *velocities_input[NFLUIDS] = {"
        for i in range(nfluids):
            if i == nfluids-1:
                newline += "v_input"+str(i)
            else:
                newline += "v_input"+str(i)+","
        newline += "};\n"
        ofile.write(newline)
        newline = "real *velocities_output[NFLUIDS] = {"
        for i in range(nfluids):
            if i == nfluids-1:
                newline += "v_output"+str(i)
            else:
                newline += "v_output"+str(i)+","
        newline += "};\n"
        ofile.write(newline)
        continue

    if re.search("%FLUIDS2",line):
        newline = ""
        for i in range(nfluids):
            newline += "rho[{:d}],\n".format(i)
            newline += "velocities_input[{:d}],\n".format(i)
            if i == nfluids-1:
                newline += "velocities_output[{:d}]);\n".format(i)
            else:
                newline += "velocities_output[{:d}],\n".format(i)
        ofile.write(newline)
        continue

    ofile.write(line)
