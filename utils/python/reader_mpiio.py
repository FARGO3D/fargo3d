from numpy import fromfile

class Fields():
    def __init__(self, directory, fluid, n, dtype="float64"):
        """Reading FARGO3D parallel outputs (.mpiio) files.
        
        The outputgas.dat/outputdust.dat files are assumed to be in the
        same directory as the data files.
        
        Reading data stored in "directory". The data file corresponds to
        the fluid "fluid" at output number "n". The data type is given by
        the "dtype" argument. Possible options are "float64" or
        "float32".
        
        Note: The instantation of the class does not read the file. To get
        a field, call the method get_field(fieldname).
        
        Example:
        
        f = Fields("outputs/fargo","gas",0)
        density = f.get_field("dens").reshape(ny,nx)

        """
        if len(directory)>1:
            directory = directory.strip() #We remove possible spaces
            if directory[-1] != "/": directory += "/"
        self.name      = fluid
        self.filename  = fluid + "_" + str(n)+".mpiio"
        self.directory = directory
        self.dtype     = dtype
        self.ifile     = open(directory+self.filename,"r")
        self.offsets, self.count = self.parse_offsets_file(directory)
        if dtype == "float64":
            self.dtype_int = 8
        else:
            self.dtype = 4
        
    def parse_offsets_file(self,directory):
        offsets = open(directory+"output{:s}.dat".format(self.name))
        dict_offset = {}
        for i,line in enumerate(offsets.readlines()):
            offset   = int(line.split()[0])
            fullname = line.split()[1]
            dict_offset[fullname] = offset
            if i == 0:
                count = -1
            else: #Assuming that all fields have the same count...
                count = offset-old_offset
            old_offset = offset
        offsets.close()
        return dict_offset,count
    
    def get_field(self, fieldname):
        if fieldname not in["bx","by","bz","divb"]:
            self.ifile.seek(self.offsets[self.name+fieldname]*self.dtype_int)
        else:
            self.ifile.seek(self.offsets[fieldname]*self.dtype_int)
        return fromfile(self.ifile,count=self.count, dtype=self.dtype)
