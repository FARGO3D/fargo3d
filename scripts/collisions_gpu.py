import re
import argparse
import textwrap

parser = argparse.ArgumentParser(prog='collisions_gpu.py',
    description='Compile GPU collisions template to kernel.',
    epilog='This script is part of FARGO3D.')
parser.add_argument('input')
parser.add_argument('-n', '--nfluids', default=1, type=int)
parser.add_argument('-o', '--output', default='collisions.cu')
args = parser.parse_args()

with open(args.input, 'r') as template_file, \
        open(args.output, 'w') as kernel_file:
    # read the entire template file
    template = template_file.read()

    # form the function arguments (type + name)
    entries = list()
    for i in range(args.nfluids):
        entries.append(f'real *rho{i:d}')
        entries.append(f'real *v_input{i:d}')
        entries.append(f'real *v_output{i:d}')
    fluids_defs = ', '.join(entries)

    # form the input arguments (name + index)
    entries = list()
    for i in range(args.nfluids):
        entries.append(f'rho[{i:d}]')
        entries.append(f'velocities_input[{i:d}]')
        entries.append(f'velocities_output[{i:d}]')
    fluids_inputs = ', '.join(entries)

    # form the array assignments
    rho_assign = ',\n  '.join([f'rho{i:d}' for i in range(args.nfluids)])
    vin_assign = ',\n  '.join([f'v_input{i:d}' for i in range(args.nfluids)])
    vout_assign = ',\n  '.join([f'v_output{i:d}' for i in range(args.nfluids)])

    fluids_assign = textwrap.dedent(f'''\
        real *rho[NFLUIDS] = {{
          {rho_assign}
        }};
        real *velocities_input[NFLUIDS] = {{
          {vin_assign}
        }};
        real *velocities_output[NFLUIDS] = {{
          {vout_assign}
        }};
        ''')

    kernel_file.write(template.format(fluids_defs=fluids_defs,
        fluids_inputs=fluids_inputs, fluids_assign=fluids_assign))
