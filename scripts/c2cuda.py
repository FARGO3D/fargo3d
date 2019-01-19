#!/usr/bin/env python

"""
C2CUDA parser developped for the FARGO3D code (http://fargo.in2p3.fr)
Pablo Benitez Llambay, 2012-2014
"""

import re
import sys
import getopt
import os

def verb(ifile, ofile):
    print '\nVERBOSE MODE ACTIVATED'
    print '=======================\n'
    print '\nInput file: ', ifile
    print 'Output file:',  ofile

def read_file(input_file):
    try:
        ifile = open(input_file,'r')
    except IOError:
        print '\nI/O error in c2cuda.py! Please, verify your input/output files.\n'
        exit()
    return ifile.readlines()

def usage():
    print '\nUsage: -i --input=  --> input_file'
    print '       -o --output= --> output file'
    print '       -v --verbose --> verbose mode'
    print '       -f --formatted --> formatted with astyle (external dependence)'
    print '       -p --profiling --> for block-dim studies.\n'
    print '       -s --setup --> setup name.\n'

    exit()

def opt_reader():
    #default values:
    verbose = False
    formated = False
       
    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           'i:o:s:vfp',
                                           ['input=',
                                            'output=',
                                            'verbose',
                                            'formated',
                                            'profiling',
                                            'setup='])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        
    if(options == []):
        usage()

    o_file = i_file = ''

    global profiling
    global SETUP
    global INPUT

    SETUP = ""
    profiling = False
            
    for opt,arg in options:
        if opt in ('-o', '--output'):
            o_file = arg
            continue
        if opt in ('-i', '--input'):
            i_file = arg
            INPUT = arg
            continue
        if opt in ('-v', '--verbose'):
            verbose = True
            continue
        if opt in ('-f', '--formated'):
            formated = True
        if opt in ('-p', '--profiling'):
            profiling = True
            continue
        if opt in ('-s', '--setup'):
            SETUP=arg
            continue

    opt = {'verbose': verbose,
           'input': i_file,
           'output': o_file,
           'formated': formated,
           'profiling': profiling,
           'setup':SETUP}

    return opt
        
def literal(lines, option, verbose = False):

    founded = False
    output = []

    begin = '//<'   + option + '>'
    end   = '//<\\' + option + '>'
    
    if verbose:
        print '\n---------------------------------'
        print 'Looking for ', option, ' lines.\n'

    for line in lines:
        line = line[:-1] # Avoiding \n
        if line == begin:
            founded = True
            continue

        if line == end:
            if verbose:
                if output == []:
                    print option, ' is empty...'
                print '\nAll ' + option +  ' lines were stored.'
                print '---------------------------------\n'
            return output

        if founded:
            output.append(line)
            if verbose:
                print line[:-1], 'is a/an ' + option + ' line.'

def main_func(lines, verbose=False, test=False):

    if verbose:
        print '\n---------------------------------'
        print 'Searching cpu main function...\n'
    
    function = re.compile(r"""
               (\w+)         #function type "1"
               \s+           #1 or more whitespace
               (\w+)_cpu     #function name "2"
               (\s?|\s+)     #1 or more whitespace
               \( (.*) \)    #input variables (all of them) "4"
               """, re.VERBOSE)

    if test:
        print
        print 'TEST OF MAIN_FUNC'
        print '================='
        print
        
        test_lines = ['void function_cpu (real dt, float b, string str_1) {',
                      'void function_cpu (real dt, float b){',
                      'void function_cpu(real dt, float b){',
                      'void function_cpu(real dt, float b, int j) {',
                      'void function_cpu (real dt,float b) {',
                      'void function_cpu(real dt,float b){',
                      'void     function_cpu    (real dt,   float b)    {']
        for line in test_lines:
            s = function.search(line)     
            if(s):
                func_type = s.group(1)
                func_name = s.group(2)
                func_var  = re.sub(',(\s+|\s?)',', ',s.group(4))
                parsed_line = func_type + ' ' + func_name
                parsed_line += '_gpu' + '(' + func_var + ') {\n'
                print line, " was parsed as "
                print parsed_line

        exit()
    for line in lines:
        s = function.search(line)
        if(s):
            func_type = s.group(1)
            func_name = s.group(2)
            func_var  = re.sub(',(\s+|\s?)',', ',s.group(4)) 
            parsed_line = func_type + ' ' + func_name
            parsed_line += '_gpu' + '(' + func_var + ') {\n'
            if(verbose):
                print line[:-1], " was parsed as "
                print parsed_line
                print "Function", func_name, "was found."
                print '---------------------------------'
            return parsed_line

def gathering_data(lines,verbose):

    flags     = literal(lines,'FLAGS',verbose)
    includes  = literal(lines,'INCLUDES',verbose)
    user_def  = literal(lines,'USER_DEFINED',verbose)
    loop      = literal(lines,'LOOP_VARIABLES',verbose)
    external  = literal(lines,'EXTERNAL',verbose)
    variables = literal(lines,'VARIABLES',verbose)
    filling   = literal(lines,'FILLING_VARIABLES',verbose)
    internal  = literal(lines,'INTERNAL',verbose)
    main_loop = literal(lines,'MAIN_LOOP',verbose)
    constant  = literal(lines,'CONSTANT',verbose)
    last_block = literal(lines, 'LAST_BLOCK',verbose)
    gpu_func  = main_func(lines,test=False,verbose=verbose)
    
    data = {'flags':flags, 'includes':includes,'user_def':user_def,
            'loop':loop,'external':external,'variables':variables,
            'filling':filling,'main_loop':main_loop,'gpu_func':gpu_func,
            'constant':constant, 'internal':internal, 'last_block':last_block}
    return data

def make_flags(flags):
    new_flags = []
    for element in flags:
        new_flags.append(element[2:])
    return new_flags

def parsing_external(external):
    
    ifdef_level=0
    declarations = []
    calls = []
    variables = re.compile(r"""
                \s+                #1 or more whitespace
                (\w+\*?)           #variable type "1"
                \s+                #1 or more whitespace
                (\w+)              #variable name "2"
                (\s?|\s+)=         #1 or more whitespace and = "3"
                (\s?|\s+)          #1 or more whitespace and = "4"
                (.*);              #asign arguments "5"
                """, re.VERBOSE)
    
    externals = []

    for element in external:
        if not re.search("\s?//",element): #Avoiding comments...
            if element[0] == '#':
                declarations.append(element)
                calls.append(element)
                continue
            s = variables.search(element)
            declarations.append(s.group(1)+' '+s.group(2))
            if(re.match(".*_cpu",s.group(5))):
                calls.append(re.sub("_cpu","_gpu",s.group(5)))
            else:
                calls.append(s.group(5))

        externals.append([s.group(2), s.group(5)])

    return declarations, calls, externals

def make_launcher(gpu_func,calls):

    launcher = re.search("\w*\s*(.*)",gpu_func).group(1) #avoiding type
    
    func_name = re.search("(.*)\(",launcher).group(1)

    variables = re.search("\((.*)\)",launcher).group(1).split(',')
    var = []
    for i in variables:
        if re.search('Field',i):
            continue
        try:
            var.append(i.split()[1])
        except IndexError:
            continue
    launcher = re.sub("_gpu","_kernel<<<grid,block>>>",func_name) + '('
    for i in var:
        launcher += i + ',\n'

    for element in calls:
        if(element[0] == '#'):
            launcher += element + '\n'
        else:
            launcher += element + ',\n'

    return launcher[:-2] + ") {\n", 'extern "C" ' + gpu_func
    
def make_kernel(gpu_func,declarations):
    launcher = re.sub("_gpu","_kernel",gpu_func)        
    if re.search("\(\s*\)",launcher):
#Avoiding problems if there is no argument.
        launcher = "__global__ " + launcher[:-4] + '\n'
    else:
        launcher = "__global__ " + launcher[:-4] + ',\n'
    """
    Note that this line implies that any string
    defined after some Field string will be removed.
    So, in order to avoid problems, you must define
    all the Field variables at the end of the function!!!
    """
    temporal1 = re.search("\(.*",launcher).group(0)
    temporal1 = re.sub("Field.*"," ",temporal1)
    launcher = re.search("(.*)\(",launcher).group(1) + \
        temporal1   
    for element in declarations:
        if(element[0] == '#'):
            launcher += element + '\n'
        else:
            launcher += element + ',\n'
    return launcher[:-2] + ") {\n"


def make_constant(symbols):
    
    data = re.compile(r"""
               (\s?|\s+)//       #comments "1"
               (\s?|\s+)(\w+)    #type "3"
               (\s?|\s+)(\w+)    #name "5"
               (\s?|\s+)\(?      #whitespace? "6"
               (.*)\);           #size "7"               
               """, re.VERBOSE)

    cte_cpy = ''
    sizes = []

    for line in symbols:
        s = data.search(line)
        try:
            size = int(s.group(7)) #if it is a number
        except ValueError:
            size = 796; #any value!
        if(size == 1):
            cte_cpy += 'cudaMemcpyToSymbol(' + \
                 s.group(5) + '_s, ' + \
                "&" + s.group(5) + ', ' + \
                'sizeof(' + s.group(3) + ')' + \
                '*(' + s.group(7) + '), ' + \
                '0, cudaMemcpyHostToDevice);\n'
        else:            
            cte_cpy += 'CUDAMEMCPY(' + \
                s.group(5) + '_s, ' + \
                s.group(5) + '_d, ' + \
                'sizeof(' + s.group(3) + ')' + \
                '*(' + s.group(7) + '), ' + \
                '0, cudaMemcpyDeviceToDevice);\n'
        sizes.append(s.group(7))

#determining size of constant memory....
    numvar = len(symbols)
    exact_size = 0
    vectors = 0
    for i in sizes:
        try:            
            exact_size +=int(i)
        except ValueError:
            vectors += 1
    try:
        vector_size = int((15384/2-exact_size)/vectors)
    except ZeroDivisionError:
        vector_size = 0
    
    cte_dec = ''
    defines = ''
    undefs  = ''

    for line in symbols:
        s = data.search(line)
        try:
            size = int(s.group(7)) #if it is a number
        except ValueError:
            size = vector_size
        if(size > 1):
            cte_dec += 'CONSTANT(' + \
                s.group(3)+ ', ' + \
                s.group(5) + '_s, ' + str(size) + ');\n'
            defines += '#define ' + s.group(5) + "(i) " + s.group(5) + "_s[(i)]\n" 
        else:
            cte_dec += '__device__ __constant__ ' + \
                s.group(3) + ' ' + \
                s.group(5) + '_s;\n'
            defines += '#define ' + s.group(5) + " " + s.group(5) + "_s\n"
            undefs  += '#undef ' + s.group(5) + '\n'

    return cte_cpy, cte_dec, defines, undefs


def make_mainloop(mainloop):
    data = re.compile(r"""
               (\s?|\s+)for\s*\(                 #identifying a for "1"
               (\s?|\s+)(.*)(\s+|\s?)=        #ivariable "3"
               (\s?|\s+)(.*)(\s+|\s?);        #lower index "6"
               (\s?|\s+).*(\s?|\s+)<
               (\s?|\s+)(.*)(\s?|\s+);        #upper index 11
               """, re.VERBOSE)
    var = []

    loop = False
    
    begin = '//<'   + '#' + '>'
    end   = '//<\\' + '#' + '>'

    effective_loop = []

    for line in mainloop:
        if data.search(line):
            s = data.search(line)
            var.append([s.group(3),s.group(6),s.group(11)])
        if line == begin:
            loop = True
            continue
        if line == end:
            loop = False
            break
        if loop:
            effective_loop.append(line)
    second_line = ''
    for i in var:
        first_line = 'i = threadIdx.x + blockIdx.x * blockDim.x;\n' + \
            'j = threadIdx.y + blockIdx.y * blockDim.y;\n' + \
            'k = threadIdx.z + blockIdx.z * blockDim.z;\n'
        second_line += i[0] + '>=' + i[1] + \
            ' && ' + i[0] + '<' + i[2] + ' && '

    second_line = ''

    first_line = '#ifdef X \n' + \
        'i = threadIdx.x + blockIdx.x * blockDim.x;\n' + \
        '#else \n' + \
        'i = 0;\n' + \
        '#endif \n' + \
        '#ifdef Y \n' + \
        'j = threadIdx.y + blockIdx.y * blockDim.y;\n' + \
        '#else \n' + \
        'j = 0;\n' + \
        '#endif \n' + \
        '#ifdef Z \n' + \
        'k = threadIdx.z + blockIdx.z * blockDim.z;\n' + \
        '#else \n' + \
        'k = 0;\n' + \
        '#endif\n'
    second_line += '#ifdef Z\n'
    second_line += 'if(' + var[0][0] + '>=' + var[0][1] + \
        ' && ' + var[0][0] + '<' + var[0][2] + ') {\n'
    second_line += '#endif\n'
    second_line += '#ifdef Y\n'
    second_line += 'if(' + var[1][0] + '>=' + var[1][1] + \
        ' && ' + var[1][0] + '<' + var[1][2] + ') {\n'
    second_line += '#endif\n'    
    second_line += '#ifdef X\n'
    second_line += 'if(' + var[2][0] + '<' + var[2][2] + ') {\n'
    second_line += '#endif\n'    

    return first_line, second_line, effective_loop, var

def output(data):

    out = '//This file was created automatically by the script c2cuda.py\n'

    for element in data['includes']: #Writing includes
        out += element + '\n'

    out += data['gpu_func'] #Writing the launcher
    print out

def make_topology(var_loop, externals):

    blocks_define = '' 

    BLOCKS = analyze_blocks()
    if(BLOCKS != None):
        blocks = 'dim3 block ({0:s}, {1:s}, {2:s});'.format(BLOCKS[0],BLOCKS[1],BLOCKS[2])
    else:
        blocks = 'dim3 block (BLOCK_X, BLOCK_Y, BLOCK_Z);'

    size_x = size_y = size_z = 0

    #Matching loop variables with external variables
    for varex,asign in externals:
        for index,minval,maxval in var_loop:            
            if varex == maxval:
                if index == 'i':
                    size_x = asign
                elif index == 'j':
                    size_y = asign
                elif index == 'k':
                    size_z = asign

    grid   = 'dim3 grid ((' + "Nx+2*NGHX" + '+block.x-1)/block.x,\n' + \
        '((' + "Ny+2*NGHY" + ')+block.y-1)/block.y,\n' + \
        '((' + "Nz+2*NGHZ" + ')+block.z-1)/block.z);'


    return blocks_define, blocks, grid



def analyze_blocks():
    try:
        blocks = open("../setups/"+SETUP+"/"+SETUP+".blocks","r")
    except IOError:
        return None
    for line in blocks.readlines():        
        split = line.split()
        search = re.search(split[0],INPUT[:-2])
        if search:
            BLOCK_X = split[1]
            BLOCK_Y = split[2]
            BLOCK_Z = split[3]
            return BLOCK_X,BLOCK_Y,BLOCK_Z
    return None

def make_output(f,output_file, formated=False):
    output = ''
    for line in f['flags']:
        output += line + '\n'
    
    output += '\n'
    for line in f['includes']:
        output += line + '\n'

    output += '\n' + f['defines']
#    output += '\n' + f['blocks_define']
    
    output += '\n' + f['cte_dec']
    
    output += '\n' + f['kernel'] + '\n'

    for line in f['internal']:
        output += line + '\n'

    output += '\n' + f['first_line']
    output += '\n' + f['second_line']
    
    for line in f['effective_loop']:
        output += line + '\n'
    
    output += '#ifdef X \n } \n #endif\n'
    output += '#ifdef Y \n } \n #endif\n'
    output += '#ifdef Z \n } \n #endif\n'

    output += '}\n'

    output += '\n' + f['def_launcher']

    output +=  '\n' + f['undefs'] + '\n'

    try:
        for line in f['user_def']:
            output += line + '\n'
    except TypeError:
        pass

    output += '\n' + f['blocks']
    output += '\n' + f['grid'] + '\n'

    output += '\n#ifdef BIGMEM\n'
    output += ('#define xmin_d &Xmin_d\n' + \
                   '#define ymin_d &Ymin_d\n' + \
                   '#define zmin_d &Zmin_d\n')
    output += ('#define Sxj_d &Sxj_d\n'+ \
                   '#define Syj_d &Syj_d\n'+ \
                   '#define Szj_d &Szj_d\n'+ \
                   '#define Sxk_d &Sxk_d\n'+ \
                   '#define Syk_d &Syk_d\n'+ \
                   '#define Szk_d &Szk_d\n'+ \
                   '#define InvVj_d &InvVj_d\n')
    output += '#endif\n'

    output += '\n'+ f['cte_cpy'] + '\n'

    kernel_name = re.search("(.*)<<<",f['launcher']).group(1)
    cache = 'cudaFuncSetCacheConfig(' + kernel_name + \
        ', cudaFuncCachePreferL1 );'
    output += '\n' + cache

#=================================================================
    if profiling:
        prof = """
cudaEvent_t start, stop;
float time;

cudaEventCreate(&start);
cudaEventCreate(&stop);
 
int eex, ey, ez;

#ifndef X
{ex=0;
#else
for (eex=3; eex < 7; eex++) {
#endif
#ifndef Y
{ey=0;
#else
for (ey=0; ey < 7; ey++) {
#endif
#ifndef Z
{ez=0;
#else
for (ez=0; ez < 7; ez++) {
#endif
  block.x = 1<<eex;
  block.y = 1<<ey;
  block.z = 1<<ez;
  if (block.x * block.y * block.z <= 1024) {
 grid.x  = (Nx+2*NGHX+block.x-1)/block.x;
 grid.y  = ((Ny+2*NGHY)+block.y-1)/block.y;
 grid.z  = ((Nz+2*NGHZ)+block.z-1)/block.z;

 cudaEventRecord(start, 0);


"""
        output += '\n' + prof
#=================================================================
    
    output += '\n' + f['launcher'][:-3] + ';\n'
    if not profiling:
        output += '\n' + 'check_errors("' + kernel_name + '");\n'

#=================================================================
    if profiling:

        prof = """
cudaDeviceSynchronize();
cudaError_t  cudaError = cudaGetLastError();
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
if (cudaError == 0) {
"""
        output += prof
        prof = """
cudaEventElapsedTime(&time, start, stop);
printf ("{:s}\\t%d\\t%d\\t%d\\t%f\\n", block.x, block.y, block.z, time);
""".format(output_file)

        output += '\n' + prof + "}"

        output += '}}}}' + '\nexit(1);\n'
#=================================================================

    try:
        for line in f['last_block']:
            output += line +'\n'
    except TypeError:
        pass

    output += '\n}'
    
    if(len(output_file)==0):
        print output
    else:
        out = open(output_file,'w')
        out.write(output)
    if(formated == True):
        os.system('astyle ' + output_file + "&")
    return

def main():

    options = opt_reader()

    verbose     = options['verbose']
    input_file  = options['input']
    output_file = options['output']
    
    if input_file == output_file:
        print "\nWARNING!!! You would overwrite your input file!!!"
        print "            This is not allowed...\n"
        exit()

    if(output_file[-3:] != '.cu'):
        print '\nWARNING!!! Your output file must be a CUDA file (.cu extension)!!!\n'
        exit()

    if verbose:
        verb(input_file, output_file)
    input_lines = read_file(input_file) #Reading file

    data = gathering_data(input_lines,verbose)
    
    declarations, calls,externals  = parsing_external(data['external'])
    flags                          = make_flags(data['flags'])

    launcher,def_launcher          = make_launcher(data['gpu_func'], calls)

    kernel                         = make_kernel(data['gpu_func'], declarations)

    if(data['constant']) != None:
        cte_cpy,cte_dec,defines,undefs = make_constant(data['constant'])
    else:
        cte_cpy = cte_dec = defines = undefs = ''
    
    first_line, second_line, effective_loop, var_loop \
        = make_mainloop(data['main_loop'])

    blocks_define, blocks, grid = make_topology(var_loop,externals)

    final = {'flags':flags, 'launcher':launcher, 'def_launcher':def_launcher,
             'kernel':kernel, 'includes':data['includes'], 'defines':defines,
             'internal':data['internal'], 'var_loop':var_loop,
             'user_def':data['user_def'],'filling':data['filling'], 
             'cte_cpy':cte_cpy,'cte_dec':cte_dec, 'first_line':first_line,
             'second_line':second_line, 'effective_loop':effective_loop,
             'blocks_define':blocks_define, 'blocks':blocks,'grid':grid,
             'undefs':undefs, 'last_block':data['last_block']}

    output = make_output(final, output_file, formated=options['formated'])

    #    print output_file, 'was created from', input_file

if __name__=='__main__':
    main()
