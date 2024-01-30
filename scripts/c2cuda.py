'''
C2CUDA parser developped for the FARGO3D code (http://fargo.in2p3.fr)
Pablo Benitez Llambay, 2012-2014
Modified by Ellen M. Price, 2024
'''

import os
import re
import argparse
import textwrap
from collections import namedtuple

parser = argparse.ArgumentParser(prog='c2cuda.py',
    description='Compile structured .c file to .cu',
    epilog='This script is part of FARGO3D.')
parser.add_argument('input', nargs='+')
parser.add_argument('-o', '--outdir', default='')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-p', '--profiling', action='store_true')
parser.add_argument('-b', '--blocks', default='fargo.blocks')
args = parser.parse_args()

Function = namedtuple('Function', ['return_type', 'name', 'argument_list'])
Variable = namedtuple('Variable', ['data_type', 'symbol'])
External = namedtuple('External', ['data_type', 'name', 'assignment'])
Constant = namedtuple('Constant', ['data_type', 'size', 'symbol'])

divider = '-' * 20
fill_kw = dict(initial_indent='', subsequent_indent='  ',
    break_long_words=False, break_on_hyphens=False)
fill_kw2 = dict(initial_indent='', subsequent_indent='    ',
    break_long_words=False, break_on_hyphens=False)

################################################################################
# PARSE SECTION
################################################################################

def find_block(content, tag):
    '''Find a block delimited by //<tag> and //<\\tag> in the given
    string, returning a list of the lines between the delimiters.
    '''
    rgx = r'\/\/<' + tag + r'>((?:.*\n)+?)\/\/<\\' + tag + r'>'
    pat = re.compile(rgx)

    if args.verbose:
        print(f'\n{divider}\nLooking for {tag} lines.')

    m = pat.search(content)
    if m is not None:
        lines = m.group(1)
        if args.verbose:
            n = len(lines.split('\n'))
            print(f'Found {n} {tag} lines.')
        return lines
    else:
        return ''

def parse_main_func(content):
    '''Finds a function declaration in the input and returns the
    transformed version for GPU.
    '''
    fn = None

    if args.verbose:
        print(f'\n{divider}\nParsing main CPU function.')

    # This regex will match a C function definition like:
    #   void foo_cpu(real x, real y)
    # Groups extracted:
    #   1 = return type
    #   2 = name
    #   3 = argument list
    rgx1 = r'(\w+)\s+(\w+)_cpu\s*\((.*)\)'
    pat1 = re.compile(rgx1)

    rgx2 = r'(\w+\s*\*?\s*)(\w+)\s*,?'
    pat2 = re.compile(rgx2)

    m = pat1.search(content)
    if m is not None:
        before = m.group(0)
        rtype, fname, arglist = m.group(1, 2, 3)

        parsed_arglist = [Variable(data_type=m2.group(1), symbol=m2.group(2)) \
            for m2 in pat2.finditer(arglist)]

        # Construct a new function definition
        fn = Function(return_type=rtype, name=fname, argument_list=parsed_arglist)

        if args.verbose:
            after = f'{rtype} {fname}({arglist})'
            print(textwrap.dedent(f'''
                Found function {name}:
                {before} => {after}
                {divider}
                '''))

    return fn

def parse_external(block_content):
    '''Parses the external block for variable assignments, extracting the
    type, variable name, and right-hand side assignment.
    '''
    external = list()

    # This regex will match a C variable assignment like:
    #   real* foo = bar;
    # Groups extracted:
    #   1 = variable type
    #   2 = lhs name
    #   3 = rhs value
    assign = re.compile(r'\s*(\w+\s*\*?\s*)(\w+)\s*=\s*(.*);')

    for line in block_content.split('\n'):
        if len(line) == 0: continue
        elif line[0] == '#':
            # Preserve preprocessor directives verbatim
            external.append('\n' + line + '\n')
        else:
            # Loop over all matches on this line
            for m in assign.finditer(line):
                dtype, name, rhs = m.group(1, 2, 3)
                rhs = rhs.replace('_cpu', '_gpu')
                ext = External(data_type=dtype, name=name, assignment=rhs)
                external.append(ext)

    return external

def parse_constant(block_content):
    '''Find all instances of constant memory specs in the given content;
    returns a list of Constant tuples.
    '''
    constant = list()

    # This regex will match a constant spec like:
    #   real foo(16);
    # Groups extracted:
    #   1 = variable type
    #   2 = symbol name
    #   3 = size in elements
    pat = re.compile(r'\s*\/\/\s*(\w+)\s*(\w+)\s*\((.*)\);')

    for m in pat.finditer(block_content):
        dtype, name, size = m.group(1, 2, 3)

        try: size = int(size)
        except ValueError: pass

        constant.append(Constant(data_type=dtype, symbol=name, size=size))

    return constant

def parse_topology(gpu_func):
    '''Open the blocks file and extract the entries for
    BLOCK_X, BLOCK_Y, BLOCK_Z, returned as a tuple
    '''
    try:
        with open(args.blocks, 'r') as blocks_file:
            for line in blocks_file:
                split = line.split()
                m = re.search(split[0], gpu_func.name)
                if m is not None:
                    return split[1:4]
    except IOError:
        pass
    return None

def gather_data(content):
    '''Finds all blocks recognized by this script, returning the blob
    of data as a dictionary.
    '''
    flags      = find_block(content, 'FLAGS')
    includes   = find_block(content, 'INCLUDES')
    user_def   = find_block(content, 'USER_DEFINED')
    loop       = find_block(content, 'LOOP_VARIABLES')
    external   = find_block(content, 'EXTERNAL')
    variables  = find_block(content, 'VARIABLES')
    filling    = find_block(content, 'FILLING_VARIABLES')
    internal   = find_block(content, 'INTERNAL')
    main_loop  = find_block(content, 'MAIN_LOOP')
    constant   = find_block(content, 'CONSTANT')
    last_block = find_block(content, 'LAST_BLOCK')

    loop_body  = find_block(main_loop, '#')

    gpu_func = parse_main_func(content)
    external = parse_external(external)
    constant = parse_constant(constant)
    topology = parse_topology(gpu_func)

    data = dict(flags=flags, includes=includes, user_def=user_def,
                loop=loop, external=external, variables=variables,
                filling=filling, internal=internal, main_loop=main_loop,
                loop_body=loop_body, constant=constant, last_block=last_block,
                gpu_func=gpu_func, topology=topology)
    return data

################################################################################
# BUILD SECTION
################################################################################

def make_flags(data):
    '''Strips the comment delimiter from the beginning of each flag
    line, so they can be added to the GPU source file.
    '''
    flags_lines = list()
    pat = re.compile('\/\/(.*)')
    return '\n'.join([m.group(1) for m in pat.finditer(data['flags'])])

def make_constant(constant):
    '''Generates lines of code needed to copy constant symbols into
    device memory. Returns four lists of lines for copies, declarations,
    defines, and undefines, in that order.
    '''
    copy_lines = list()
    declare_lines = list()
    define_lines = list()
    undef_lines = list()

    for cnst in constant:
        if cnst.size == 1:
            # scalar constant
            copy_lines.append(textwrap.fill(textwrap.dedent(f'''\
                cudaMemcpyToSymbol({cnst.symbol}_s, &{cnst.symbol},
                sizeof({cnst.data_type}), 0,
                cudaMemcpyHostToDevice);'''), **fill_kw))
        else:
            # non-scalar constant
            copy_lines.append(textwrap.fill(textwrap.dedent(f'''\
                CUDAMEMCPY({cnst.symbol}_s, {cnst.symbol}_d,
                sizeof({cnst.data_type}) * ({cnst.size}),
                0, cudaMemcpyDeviceToDevice);'''), **fill_kw))

    # add up the known sizes
    exact_size = sum([cnst.size for cnst in constant \
        if isinstance(cnst.size, int)])

    # TODO: this was previously hardcoded to 15384/2, but it's not clear
    # why. If this number had some significance, it would be good to
    # document that here. 64 kib is based on the CUDA standard.
    max_const_mem  = 8192    # 64 kib / 64 bits (most conservative)
    max_const_mem -= exact_size

    nvectors = sum([1 for cnst in constant if not isinstance(cnst.size, int)])
    vector_size = int(max_const_mem / nvectors) if nvectors > 0 else 0

    for cnst in constant:
        size = cnst.size if isinstance(cnst.size, int) else vector_size
        if size > 1:
            declare_lines.append(f'''\
                CONSTANT({cnst.data_type}, {cnst.symbol}_s, {size:d});''')
            define_lines.append(f'''\
                #define {cnst.symbol}(i) {cnst.symbol}_s[(i)]''')
        else:
            declare_lines.append(f'''\
                __device__ __constant__ {cnst.data_type} {cnst.symbol}_s;''')
            define_lines.append(f'''#define {cnst.symbol} {cnst.symbol}_s''')
            undef_lines.append(f'''#undef {cnst.symbol}''')

    copy_lines = textwrap.dedent('\n'.join(copy_lines))
    declare_lines = textwrap.dedent('\n'.join(declare_lines))
    define_lines = textwrap.dedent('\n'.join(define_lines))
    undef_lines = textwrap.dedent('\n'.join(undef_lines))

    return copy_lines, declare_lines, define_lines, undef_lines

def make_launcher(data):
    '''Constructs the kernel launch call. If topology is given, the dimensions
    are hardcoded here; otherwise, use the macros defined in the code.
    '''
    gpu_func = data['gpu_func']
    external = data['external']
    constant = data['constant']
    topology = data['topology']
    user_def = data['user_def']
    last_block = data['last_block']

    copy_lines, *_ = make_constant(constant)

    if topology is not None:
        blocks_def = 'dim3 block({:d}, {:d}, {:d});'.format(*topology)
    else:
        blocks_def = 'dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);'

    # Form launch argument list as a string
    launch_args = ''
    for arg in gpu_func.argument_list:
        if isinstance(arg, Variable):
            launch_args += f'{arg.data_type} {arg.symbol}, '
        else:
            launch_args += arg
    launch_args = launch_args.strip(' ,')

    # Form kernel argument list as a string
    kernel_args = ''.join([f'{arg.symbol}, ' \
        for arg in gpu_func.argument_list if 'Field' not in arg.data_type])
    for ext in external:
        if isinstance(ext, External):
            kernel_args += ext.assignment + ', '
        else:
            kernel_args += ext
    kernel_args = kernel_args.strip(' ,')

    cache_line = f'cudaFuncSetCacheConfig({gpu_func.name}_kernel, cudaFuncCachePreferL1);'
    kernel_line = f'{gpu_func.name}_kernel<<<grid, block>>>({kernel_args});'

    output  = ''
    output += textwrap.dedent(f'''\
        extern "C"
        {gpu_func.return_type} {gpu_func.name}_gpu({launch_args}) {{
        ''')
    output += user_def + '\n'
    output += textwrap.dedent(f'''\
          {blocks_def}
          dim3 grid(((Nx+2*NGHX)+block.x-1)/block.x,
                    ((Ny+2*NGHY)+block.y-1)/block.y,
                    ((Nz+2*NGHZ)+block.z-1)/block.z);

        #if BIGMEM
        #define xmin_d &Xmin_d
        #define ymin_d &Ymin_d
        #define zmin_d &Zmin_d
        #define Sxj_d &Sxj_d
        #define Syj_d &Syj_d
        #define Szj_d &Szj_d
        #define Sxk_d &Sxk_d
        #define Syk_d &Syk_d
        #define Szk_d &Szk_d
        #define Sxi_d &Sxi_d
        #define InvVj_d &InvVj_d
        #define InvDiffXmed_d &InvDiffXmed_d
        #endif
        ''')
    output += '\n' + textwrap.indent(copy_lines, ' ' * 2) + '\n'
    output += '\n' + textwrap.indent(cache_line, ' ' * 2) + '\n'
    output += kernel_line + '\n'
    output += textwrap.dedent(f'''\
          check_errors("{gpu_func.name}_kernel");
        ''')
    output += last_block + '\n'
    output += textwrap.dedent(f'''\
        }}''')
    return output

def make_profiling_launcher(data):
    gpu_func = data['gpu_func']

    return textwrap.dedent(f'''\
            cudaEvent_t start, stop;
            float time;
            dim3 block, grid;
            int eex, ey, ez;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

        #if XDIM
            for (eex = 3; eex < 7; eex++) {{
        #else
            eex = 0;
        #endif
        #if YDIM
            for (ey=0; ey < 7; ey++) {{
        #else
            ey = 0;
        #endif
        #if ZDIM
            for (ez=0; ez < 7; ez++) {{
        #else
            ez = 0;
        #endif
            block.x = 1 << eex;
            block.y = 1 << ey;
            block.z = 1 << ez;
            if (block.x * block.y * block.z <= 1024) {{
                grid.x  = ((Nx+2*NGHX)+block.x-1)/block.x;
                grid.y  = ((Ny+2*NGHY)+block.y-1)/block.y;
                grid.z  = ((Nz+2*NGHZ)+block.z-1)/block.z;

                cudaEventRecord(start, 0);

                cudaDeviceSynchronize();
                cudaError_t cudaError = cudaGetLastError();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                if (cudaError != 0) exit(EXIT_FAILURE);

                cudaEventElapsedTime(&time, start, stop);
                printf("{gpu_func.name}\\t%d\\t%d\\t%d\\t%f\\n",
                    block.x, block.y, block.z, time);
            }}
        #if XDIM
            }}
        #endif
        #if YDIM
            }}
        #endif
        #if ZDIM
            }}
        #endif
        ''')

def make_kernel(data):
    '''Constructs the CUDA kernel function, including the full loop
    body and function declaration.
    '''
    gpu_func = data['gpu_func']
    external = data['external']
    internal = data['internal']
    constant = data['constant']
    loop_body = data['loop_body']

    _, declare_lines, define_lines, undef_lines = make_constant(constant)

    # This regex will match something that looks like the beginning
    # of a C for loop:
    #   for (q = m; q < n; ...
    # Groups:
    #   1 = variable name
    #   2 = lower limit
    #   3 = upper limit
    rgx = r'\s*for\s*\(\s*(.+)\s*=\s*(.+)\s*;\s*.+\s*<\s*(.+);'
    pat = re.compile(rgx)

    # Find the loop limits and save in `vbls`
    loop_vbls = list()
    for m in pat.finditer(data['main_loop']):
        loop_vbls.append(m.group(1, 2, 3))
        if len(loop_vbls) == 3: break

    # Ensure we found the right number of loops! There are cases where more than
    # three nested loops is okay, but any loops after three aren't distributed
    # across GPU blocks
    if len(loop_vbls) < 3:
        d = len(loop_vbls)
        raise ValueError(f'Expected a nested loop of depth 3, but found depth {d}')

    kvar, klo, khi = loop_vbls[0]
    jvar, jlo, jhi = loop_vbls[1]
    ivar, _,   ihi = loop_vbls[2]

    # Form kernel argument list as a string
    kernel_args = ''.join([f'{arg.data_type} {arg.symbol}, ' \
        for arg in gpu_func.argument_list if 'Field' not in arg.data_type])
    for ext in external:
        if isinstance(ext, External):
            kernel_args += f'{ext.data_type} {ext.name}, '
        else:
            kernel_args += ext
    kernel_args = kernel_args.strip(' ,')

    output  = ''
    output += define_lines + '\n\n'
    output += declare_lines + '\n\n'
    output += textwrap.dedent(f'''\
        __global__ void {gpu_func.name}_kernel({kernel_args}) {{
        ''')
    output += internal + '\n'
    output += textwrap.dedent(f'''\
        #if XDIM
          i = threadIdx.x + blockIdx.x * blockDim.x;
        #else
          i = 0;
        #endif
        #if YDIM
          j = threadIdx.y + blockIdx.y * blockDim.y;
        #else
          j = 0;
        #endif
        #if ZDIM
          k = threadIdx.z + blockIdx.z * blockDim.z;
        #else
          k = 0;
        #endif

        #if ZDIM
          if (({kvar} >= {klo}) && ({kvar} < {khi})) {{
        #endif
        #if YDIM
            if (({jvar} >= {jlo}) && ({jvar} < {jhi})) {{
        #endif
        #if XDIM
              if ({ivar} < {ihi}) {{
        #endif
        ''')
    output += loop_body
    output += textwrap.dedent(f'''\
        #if XDIM
              }}
        #endif
        #if YDIM
            }}
        #endif
        #if ZDIM
          }}
        #endif
        }}
        ''')
    output += undef_lines + '\n'
    return output

for input_file in args.input:
    # form the output file name, raising an error if it points to
    # the same file as the input file name
    _, input_base = os.path.split(input_file)
    output_file = os.path.join(args.outdir, os.path.splitext(input_base)[0] + '.cu')
    if os.path.isfile(output_file) and os.path.samefile(input_file, output_file):
        msg = textwrap.dedent(f'''\
            Input file {input_file} and its output file {output_file} are the
            same file, but overwriting is not allowed.
            ''')
        raise ValueError(msg)

    if args.verbose:
        print(f'Processing {input_file} -> {output_file}')

    # read the entire input file
    with open(input_file, 'r') as infile:
        content = infile.read()

    data = gather_data(content)

    flags = make_flags(data)
    kernel = make_kernel(data)
    includes = data['includes']

    if args.profiling:
        launcher = make_profiling_launcher(data)
    else:
        launcher = make_launcher(data)

    output  = ''
    output += flags + '\n\n'
    output += includes + '\n\n'
    output += kernel + '\n\n'
    output += launcher

    warning = textwrap.fill(textwrap.dedent(f'''\
        /* This file was created automatically during compilation from
        {input_file}. Do not edit. See python script 'c2cuda.py' for details. */
        '''))

    with open(output_file, 'w') as outfile:
        outfile.write(warning + '\n\n')
        outfile.write(output)
