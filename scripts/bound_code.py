import sys

FLUIDNUMBER =  int(sys.argv[2])

with open("../scripts/bound_cpu.code", 'w', encoding='utf-8') as bound_cpu, \
     open("../scripts/bound_gpu.code", 'w', encoding='utf-8') as bound_gpu, \
     open("../scripts/bound_proto.code", 'w', encoding='utf-8') as bound_proto:
    for i in range(FLUIDNUMBER):
        for s in ['y','z']:
            bound_cpu.write("#ifdef "+s.upper()+"\n")
            bound_cpu.write("boundary_{0:s}min[{1:d}] = boundary_{0:s}min_{1:d}_cpu;\n".format(s,i))
            bound_cpu.write("boundary_{0:s}max[{1:d}] = boundary_{0:s}max_{1:d}_cpu;\n".format(s,i))
            bound_cpu.write("#endif\n")

            bound_gpu.write("#ifdef "+s.upper()+"\n")
            bound_gpu.write("boundary_{0:s}min[{1:d}] = boundary_{0:s}min_{1:d}_gpu;\n".format(s,i))
            bound_gpu.write("boundary_{0:s}max[{1:d}] = boundary_{0:s}max_{1:d}_gpu;\n".format(s,i))
            bound_gpu.write("#endif\n")

            bound_proto.write("#ifdef "+s.upper()+"\n")
            bound_proto.write("ex void boundary_{0:s}min_{1:d}_cpu(void);\n".format(s,i))
            bound_proto.write("ex void boundary_{0:s}max_{1:d}_cpu(void);\n".format(s,i))
            bound_proto.write("#endif\n")

            bound_proto.write("#ifdef "+s.upper()+"\n")
            bound_proto.write("ex void boundary_{0:s}min_{1:d}_gpu(void);\n".format(s,i))
            bound_proto.write("ex void boundary_{0:s}max_{1:d}_gpu(void);\n".format(s,i))
            bound_proto.write("#endif\n")
