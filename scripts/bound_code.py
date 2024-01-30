import argparse
import textwrap

parser = argparse.ArgumentParser(prog="boundparser.py",
    description="Produce CPU and GPU prototypes of boundary functions",
    epilog="This script is part of FARGO3D.")
parser.add_argument("-n", "--nfluids", type=int, default=1)
parser.add_argument("-cpu", help="Output file for CPU prototypes", default="bound_cpu.code")
parser.add_argument("-gpu", help="Output file for GPU prototypes", default="bound_gpu.code")
parser.add_argument("-proto", help="Output file for all prototypes", default="bound_proto.code")
args = parser.parse_args()

with open(args.cpu, "w") as bound_cpu, open(args.gpu, "w") as bound_gpu, \
        open(args.proto, "w") as bound_proto:

    for i in range(args.nfluids):
        for s in ['y', 'z']:
            supper = s.upper()

            lines = f"""\
                #if {supper}DIM
                boundary_{s}min[{i}] = boundary_{s}min_{i}_cpu;
                boundary_{s}max[{i}] = boundary_{s}max_{i}_cpu;
                #endif
                """
            bound_cpu.write(textwrap.dedent(lines))

            lines = f"""\
                #if {supper}DIM
                boundary_{s}min[{i}] = boundary_{s}min_{i}_gpu;
                boundary_{s}max[{i}] = boundary_{s}max_{i}_gpu;
                #endif
                """
            bound_gpu.write(textwrap.dedent(lines))

            lines = f"""\
                #if {supper}DIM
                ex void boundary_{s}min_{i}_cpu(void);
                ex void boundary_{s}max_{i}_cpu(void);
                ex void boundary_{s}min_{i}_gpu(void);
                ex void boundary_{s}max_{i}_gpu(void);
                #endif
                """
            bound_proto.write(textwrap.dedent(lines))
