import numpy as np
import inspect
try:
    import matplotlib.pyplot as plt
except ImportError:
    print "Matplotlib not found"
import os
import re

def color(name):
    colors = {}

    colors['reset']    = "\033[0m"
    colors['black']    = "\033[30m"
    colors['red']      = "\033[31m"
    colors['green']    = "\033[32m"
    colors['yellow']   = "\033[33m"
    colors['blue']     = "\033[34m"
    colors['magenta']  = "\033[35m"
    colors['cyan']     = "\033[36m"
    colors['white']    = "\033[37m"
    colors['bblack']   = "\033[1m\033[30m"
    colors['bred']     = "\033[1m\033[31m"
    colors['bgreen']   = "\033[1m\033[32m"
    colors['byellow']  = "\033[1m\033[33m"
    colors['bblue']    = "\033[1m\033[34m"
    colors['bmagenta'] = "\033[1m\033[35m"
    colors['bcyan']    = "\033[1m\033[36m"
    colors['bwhite']   = "\033[1m\033[37m"
    return colors[name]

class Array():
    def __init__(self,filename=None,array=None,name=None,dtype="float64",compact=True, skipnan=False):
        if filename != None:
            self.data = np.fromfile(filename,dtype=dtype)
        elif not(array is None):
            self.data = array
        else:
            print "You must include filename or array as argument!"
            
        prop = {}
        prop['name'] = name
        prop['min']  = self.data.min()
        prop['max']  = self.data.max()
        prop['mean'] = self.data.mean()
        prop['std']  = self.data.std()
        prop['skip'] = False

        if np.isnan(self.data).any():
            prop['skip'] = True

        self.prop = prop
        self.get_properties(compact)

    def get_properties(self, compact = True):
        if not compact:
            print "\n",self.prop['name']
            for key in self.prop:
                if key != "name":
                    print "\t" + key + ": {:3.6g}".format(self.prop[key]),

class Comparison():
    def __init__(self, file1, file2, dtype="float64", plot=False, compact=True):

        title = color('byellow') + "Comparing %s & %s:" % (file1,file2) + color('reset')
        print title
        under = ""
        for i in range(len(title))[13:]:
            under += "="
        print under
        
        self.success = True

        self.file1 = Array(filename = file1, name="File1", compact=compact)
        self.file2 = Array(filename = file2, name="File2", compact=compact)

        self.diff  = Array(array = self.file1.data-self.file2.data,
                           name="File1-File2", compact=compact)

        self.ratio = Array(array = self.file2.data/self.file1.data, 
                           name="File2/File1", compact=compact, skipnan = True)
        
        self.relative = Array(array = 
                              (self.file1.data-self.file2.data)/self.file1.data, 
                              name="(File1-File2)/File1", compact=compact) 

        print "\n"
        
        if self.diff.prop['skip'] == False:
            if self.diff.data.any():
                print "File1 and File2 differ.";
                identical = False
            else:
                print "File1 and File2 are identical.";
                identical = True
            if self.file1.prop['std']<1e-16 and self.file2.prop['std']<1e-16:
                print "Files differences are at the machine accuracy (1e-16)"
                identical = True
        else:
            print "There were NaNs in the difference."
            identical = False

        if self.ratio.prop['skip'] == False:
            mratio = self.ratio.prop['std']/self.ratio.prop['mean']
            if (mratio < 2e-12):
                print "File2/File1 has a Flat Ratio to within ",mratio
                flat = True
            else:
                print "File2/File1 is not a Flat Ratio: relative deviation = ",mratio
                self.success = self.success & False
                flat = False
        else:
            print "There were NaNs in the ratio."
            flat = False

        self.success = identical | flat

        if self.success:
            print "\n" + color('bgreen') + "TEST PASSED" + color('reset')+".\n"
        else:
            print "\n" + color('bred') + "TEST FAILED" + color('reset')+".\n"

        if plot == True:
            self.all = [self.file1,self.file2,self.diff,self.ratio,self.relative]
            self.plot()

    def plot(self):
        plt.ioff()
        fig = plt.figure(figsize=(15,3))
        axes = []
        n = len(self.all)
        for i in range(n):
            ax = fig.add_subplot(1,n,i+1)
            ax.set_title(self.all[i].prop['name'],color='w')
            ax.plot(self.all[i].data,'k')
        plt.show()


class GenericTest():
    def __init__(self,testname,
                 flags1,launch1,description1,
                 flags2,launch2,description2,
                 parfile,
                 parameters = None,
                 parameters1 = None,
                 parameters2 = None,
                 field = None,
                 plot = False,
                 dtype = 'float64',
                 verbose = False,
                 compact = True,
                 clean = True,
                 restore = True,
                 keep = True,
                 log = True,
                 n = 1):

        """
        testname --> string
           Name of the test, for the title and the main directory.
        flags1 --> string
           Compilation flags for the first instance of the test.
        launch1 --> string
           Execution line for the first instance of the code. The parfile 
           must not be included.
        Description1 --> string
           Description of the launch1 properties.
        flags2 --> string
           Compilation flags for the second instance of the test.
        launch2 --> string
           Execution line for the second instance of the code. The parfile 
           must not be included.
        Description2 --> string
           Description of the launch2 properties.
        parfile --> string
           The parfile for the test. The output directory is not important because it 
           will be modified by this program.
        parameters --> dict [None by default]
           It can be used to modify some particular parameter in the .par file without
           altering the latter.
           Example: parameters = {'nx':128,'dt':0.001,'ninterm':1}
        parameters1 --> dict [None by default]
           It can be used to modify some particular parameter in the 1st .par file without
           any explicit modification. it only has sense if parameters = None.
           Example: parameters = {'nx':128,'dt':0.001,'ninterm':1}
        parameters2 --> dict [None by default]
           It can be used to modify some particular parameter in the 2nd .par file without
           any explicit modification. It only has sense if parameters = None.
           Example: parameters = {'nx':128,'dt':0.001,'ninterm':1}
        field --> string
           If field is None, all the fields will be compared, otherwise 
           'field' will be compared. Field does not include the number of the field.
        plot --> Boolean:
           If True, you will see a plot of all the quantities for each field:
           (field1, field2, diff, ratio, (field1-field2/field1))
        dtype --> numpy.dtype string. 
           It is useful when the outputs are in single precision.
        Verbose --> Boolean
           If True, you will see all the steps. useful when you try
           to debug a specific test.
        clean --> Boolean
           If True, the output directory will be removed.
        restore --> Boolean
           If True, the code will be restored to its state prior to the test.
        keep --> Boolean
           If True you will keep the current state of the code. (opposite to restore)
        log --> Boolean
           If True one line summing up the test status is appended to a log file.
           Useful for nightly builds.
        n --> Integer
           Output number that will be compared, by default n = 1.

        """

        self.verbose = verbose
        if self.verbose:
            self.verbosity = ''
        else:
            self.verbosity = "> /dev/null 2>&1"

        self.flags1 = flags1
        self.flags2 = flags2
        self.launch1 = launch1
        self.launch2 = launch2
        self.compact = compact
        self.field = field
        self.plot = plot
        self.dtype = dtype
        self.testname = testname
        self.parfile = parfile
        self.parameters = parameters
        self.parameters1 = parameters1
        self.parameters2 = parameters2
        self.description1 = description1
        self.description2 = description2
        self.clean = clean
        self.restore = restore
        self.keep = keep
        self.log = log
        self.n = n
        self.command1 = None
        self.command2 = None
        
    def set_commands(self,command1 = None, command2 = None):
        """
        command1,2 [None] are possible sys commands after the first and second test.
        """
        self.command1 = command1
        self.command2 = command2

    def run(self):

        """
        Run the test defined.
        """           

        os.system("clear")
        title = "\nTEST: " + color('bgreen') + self.testname + color('reset')
        print title
        under = ""
        for i in range(len(title))[13:-1]:
            under += "="
        print under + "\n"

        print "Making the directory for the test... ("  + self.testname + ")"
        self.build_dir()
        if self.keep:
            print "Keeping flags state..."
            self.keep_old_status()
            print

        print color('bblue')+self.description1+color('reset')
        print "Building the first instance for the test..."
        self.compile(self.flags1)
        print "Processing parfile for the first instance"
        if self.parameters != None:
            self.process_parfile(1,self.parameters)
        else:
            self.process_parfile(1,self.parameters1)
        print "Executing the first instance..."
        self.launch(self.launch1,1)
        print

        if self.command1 != None:
            print "Executing:", color('bblue')+self.command1+color('reset'), "\n"
            os.system(self.command1)

        print color('bblue')+self.description2+color('reset')
        print "Building the second instance for the test..."
        self.compile(self.flags2)
        print "Processing parfile for the second instance"
        if self.parameters != None:
            self.process_parfile(2,self.parameters)
        else:
            self.process_parfile(2,self.parameters2)
        print "Executing the second instance..."
        self.launch(self.launch2,2)
        print 
        
        if self.command2 != None:
            print "Executing:", color('bblue')+self.command2+color('reset'), "\n"
            os.system(self.command2)
        
        print "Starting the comparison..."
        status, reldev = self.compare()
        self.status =status

        if self.restore:
            print "Restoring flags state..."
            self.restore_old_status()
            print
        if self.clean == True:
            print "Cleaning " +  self.testname + "..."
            self._clean()
        if self.log == True:
            logfile = open('tests.log', 'a')
            logfile.write (inspect.stack()[1][1]+' ('+self.testname+') \t==>\t ')
        if status:
            print color('bgreen')
            print "%%%%%%%%%%%%%%%%%%%"
            print "%   TEST PASSED   %"
            print "%%%%%%%%%%%%%%%%%%%"
            print  color('reset')
            if self.log:
                logfile.write ('PASSED ('+str(reldev)+')\n')
        else:
            print color('bred')
            print "%%%%%%%%%%%%%%%%%%%"
            print "%   TEST FAILED   %"
            print "%%%%%%%%%%%%%%%%%%%"
            print  color('reset')
            if self.log:
                logfile.write ('FAILED ('+str(reldev)+')\n')
        if self.log == True:
            logfile.close()

        print "Test done."

    def get_status(self):
        return self.status

    def compare(self):
        path1 = self.testname + "/test1/"
        path2 = self.testname + "/test2/"

        test1 = os.listdir(path1 + ".")
        test2 = os.listdir(path2 + ".")
        elements1 = []
        elements2 = []
        if self.field == None:
            search = "gas"
        else:
            search = self.field + str(self.n)
        for element in test1:
            if re.search(search,element):
                elements1.append(element)
        for element in test2:
            if re.search(search,element):
                elements2.append(element)
        elements1.sort()
        elements2.sort()

        status = True

        ThereAreFiles = False
        for element1,element2 in zip(elements1,elements2):
            ThereAreFiles = True
            C = Comparison(path1 + element1, path2 + element2, 
                           plot=self.plot,dtype=self.dtype,compact=self.compact)
            status = status & C.success
        if ThereAreFiles:
            return status, C.ratio.prop['std']/C.ratio.prop['mean']
        else:
            print color('bred')
            if elements1 == []:
                print "There was a problem with the first" + \
                    "part of the test."
            if elements2 == []:
                print "There was a problem with the second" + \
                    "part of the test."
                print "Check your test.\n" + color('reset')
            return False, -1

    def _clean(self):
        os.system("rm -fr " + self.testname)

        
    def build_dir(self):
        if os.system("mkdir "+ self.testname + self.verbosity):
            print color("bred") + "Warning!!! " + color("reset") + \
                "Directory " + self.testname + " already exist."
            while(1):
                print "Do you want to clean it? (N/y)"
                proceed = raw_input().lower()
                if proceed == "y" or proceed == "yes":
                    print "Cleaning " + self.testname + "..."
                    os.system("/bin/rm -fr " + self.testname + "/*" + self.verbosity)
                    return
                elif proceed == "n" or proceed == "no":
                    print "Please, move your " + self.testname + " directory or change" \
                        " the name of the test."
                    exit()
                else:
                    print "Do you want to clean it? (N/y)"
                    proceed = raw_input().lower()

    def process_parfile(self,number,parameters):
        par = open(self.parfile)
        new_par = open(self.testname+"/parfile"+str(number)+'.par',"w")
        for line in par.readlines():
            found = False
            if re.search("outputdir",line.lower()):
                new_par.write("OutputDir" + "\t" + self.testname + 
                              "/test" + str(number) +"\n")
                continue
            if parameters != None:
                for key in parameters:
                    if re.match("\s*" + key.lower() + "\s+",line.lower()):
                        new_par.write(key.lower() + "\t" + str(parameters[key]) + "\n")
                        found = True
                        break
                if found: continue
            new_par.write(line)
        new_par.close()

    def compile(self,flags):
        os.system("make " + flags +  self.verbosity)
        
    def launch(self, launch_string, number):
        os.system(launch_string+ " " + self.testname + "/parfile" + \
                      str(number) + ".par " +  self.verbosity)

    def restore_old_status(self):
        os.system("cp std/.lastflags.old std/.lastflags")
        os.system("make clean")
        os.system("make " +  self.verbosity)

    def keep_old_status(self):
        os.system("cp std/.lastflags std/.lastflags.old")

if __name__ == '__main__':

    description1 = """Running the Orszag-Tang vortex test with one processor."""
    description2 = """Running the Orszag-Tang vortex test with four processors."""

    MpiTest = GenericTest(testname = "MPI_TEST",
                          flags1 = "SETUP=otvortex FARGO_DISPLAY=NONE",
                          launch1 = "./fargo3d -m",
                          description1 = description1,
                          flags2 = "SETUP=otvortex FARGO_DISPLAY=NONE",
                          launch2 = "mpirun -np 4 ./fargo3d -m",
                          description2 = description2,
                          parfile = "setups/otvortex/otvortex.par",
                          verbose = False,
                          plot=False,
                          field = "Density",
                          compact = True,
                          parameters = {'dt':0.001, 'ntot':1},
                          parameters1 = None,
                          parameters2 = None)

    MpiTest.run()         #This line runs the test
