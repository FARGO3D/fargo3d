class Mesh():
    """
    Mesh class, convenient way to work the mesh data.
    Input: directory [string] -> place where the domain files are.
    """
    def __init__(self, directory=""):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            domain_x = np.loadtxt(directory+"domain_x.dat")
        except IOError:
            print "IOError with domain_x.dat"
        try:
            #We have to avoid ghost cells!
            domain_y = np.loadtxt(directory+"domain_y.dat")[3:-3]
        except IOError:
            print "IOError with domain_y.dat"
        self.xm = domain_x #X-Edge
        self.ym = domain_y #Y-Edge
        
        self.xmed = 0.5*(domain_x[:-1] + domain_x[1:]) #X-Center
        self.ymed = 0.5*(domain_y[:-1] + domain_y[1:]) #Y-Center
        
        #(Surfaces taken from the edges)
        #First we make 2D arrays for x & y, that are (theta,r)
        T,R = meshgrid(self.xm, self.ym)
        R2  = R*R
        self.surf = 0.5*(T[:-1,1:]-T[:-1,:-1])*(R2[1:,:-1]-R2[:-1,:-1])

class Parameters():
    """
    Class for reading and storing the simulation parameters.
    input: string -> name of the parfile, normally variables.par
    """
    def __init__(self, directory=''):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        try:
            params = open(directory+"variables.par",'r') #Opening the parfile
        except IOError:                  # Error checker.
            print  paramfile + " not found."
            return
        lines = params.readlines()     # Reading the parfile
        params.close()                 # Closing the parfile
        par = {}                       # Allocating a dictionary
        for line in lines:             # Iterating over the parfile
            name, value = line.split() # Spliting the name and the value (first blank)
            try:
                float(value)           # First trying with float
            except ValueError:         # If it is not float
                try:
                    int(value)         # we try with integer
                except ValueError:     # If it is not integer, we know it is string
                    value = '"' + value + '"'
            par[name] = value          # Filling the dictory
        self._params = par             # A control atribute, actually not used, good for debbuging
        for name in par:               # Iterating over the dictionary
            exec("self."+name.lower()+"="+par[name]) #Making the atributes at runtime


class Field(Mesh, Parameters):
    """
    Field class, it stores all the mesh, parameters and scalar fields.
    Input: field [string] -> filename of the field
           staggered='c' [string] -> staggered direction of the field. 
                                      Possible values: 'x', 'y', 'xy', 'yx'
           directory='' [string] -> where filename is
           dtype='float64' (numpy dtype) -> 'float64', 'float32', 
                                             depends if FARGO_OPT+=-DFLOAT is activated
    """
    def __init__(self, field, staggered='c', directory='', dtype='float64'):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
        Mesh.__init__(self, directory) #All the Mesh attributes inside Field!
        Parameters.__init__(self, directory) #All the Parameters attributes inside Field!

        #Now, the staggering:
        if staggered.count('x')>0:
            self.x = self.xm[:-1] #Do not dump the last element
        else:
            self.x = self.xmed
        if staggered.count('y')>0:
            self.y = self.ym[:-1]
        else:
            self.y = self.ymed
    
        self.data = self.__open_field(directory+field,dtype) #The scalar data is here.

    def __open_field(self, f, dtype):
        """
        Reading the data
        """
        field = fromfile(f, dtype=dtype)
        return field.reshape(self.ny, self.nx)

    def plot(self, log=False, cartesian=False, cmap='Oranges_r', **karg):
        """
        A layer to plt.imshow or pcolormesh function.
        if cartesian = True, pcolormesh is launched.
        """
        ax = gca()
        if log:
            data = np.log(self.data)
        else:
            data = self.data
        if cartesian:
            T,R = meshgrid(self.x,self.y)
            X = R*cos(T)
            Y = R*sin(T)
            pcolormesh(X,Y,data,cmap=cmap,**karg)
        else:
            ax.imshow(data, cmap = cmap, origin='lower',aspect='auto',
                      extent=[self.x[0],self.x[-1],self.y[0],self.y[-1]],
                      **karg)
            
    def contour(self, log=False, cartesian=False, **karg):
        if log:
            data = np.log(self.data)
        else:
            data = self.data
        ax = gca()
        T,R = meshgrid(self.x,self.y)
        if cartesian:
            X = R*cos(T)
            Y = R*sin(T)
            contour(X,Y,data,**karg)
        else:
            contour(T,R,data,**karg)


def shift_field(Field,direction):
    import copy
    """
    Half cell shifting along the direction provided by direction 
    direction can be ('x','y', 'xy', 'yx').
    
    After a call of this function, Field.xm/xmed has not
    sense anymore (it is not hard to improve).
    """
    F = copy.deepcopy(Field)
    if direction.count('x')>0:
        F.data = 0.5*(Field.data[:,1:]+Field.data[:,:-1])
        F.x = 0.5*(Field.x[1:]+Field.x[:-1])
    if direction.count('y')>0:
        F.data = 0.5*(F.data[1:,:]+F.data[:-1,:])
        F.y = 0.5*(F.y[1:]+F.y[:-1])

    F.nx = len(F.x)
    F.ny = len(F.y)
    
    return F


def cut_field(Field, direction, side):
    """                                                                                                  
    Cutting a field:
    Input: field --> a Field class                                                                       
           axis  --> 'x', 'y' or 'xy'                                                                    
           side  --> 'p' (plus), 'm' (minnus), 'pm' (plus/minnus)                                        
    """                                                                                                  
    import copy
    
    _cut_field = copy.deepcopy(Field)                                                                     
    ny,nx = Field.ny, Field.nx
    mx = my = px = py = 0
    
    if direction.count('x')>0:
        if side.count('m')>0:
            mx = 1                                                                                       
        if side.count('p')>0:
            px = 1
    if direction.count('y')>0:
        if side.count('m')>0:
            my = 1
        if side.count('p')>0:
            py = 1
                                                                                                         
    _cut_field.data = Field.data[my:ny-py,mx:nx-px]                                                       
    _cut_field.x = _cut_field.x[mx:nx-px]                                                                  
    _cut_field.y = _cut_field.y[my:ny-py]                                                                  
                                                                                                         
    return _cut_field

def vector_field(vx,vy, **karg):
    nsx = nsy = 3
    T,R = meshgrid(vx.x[::nsx],vx.y[::nsy])
    X = R*cos(T)
    Y = R*sin(T)
    vx = vx.data[::nsy,::nsx]
    vy = vy.data[::nsy,::nsx]
    U = vy*cos(T) - vx*sin(T)
    V = vy*sin(T) + vx*cos(T)
    ax = gca()
    ax.quiver(X,Y,U,V,scale=5,pivot='midle', **karg)

def euler(vx, vy, x, y, reverse):
    """
    Euler integrator for computing the streamlines.
    Parameters:
    ----------

    x,y: Float.
         starter coordinates.
    reverse: Boolean.
             If reverse is true, the integratin step is negative.
    Reverse inverts the sign of the velocity

    Output
    ------

    (dx,dy): (float,float).
             Are the azimutal and radial increment.
             Only works for cylindrical coordinates.
    """
    sign = 1.0
    if reverse:
        sign = -1
    vphi = get_v(vx,x,y)
    vrad = get_v(vy,x,y)
    if vphi == None or vrad == None: #Avoiding problems...                                                                                                                                                                                    
        return None,None
    l = np.min((((vx.xmax-vx.xmin)/vx.nx),((vx.ymax-vx.ymin)/vx.ny)))
    h = 0.5*l/np.sqrt((vphi**2+vrad**2))

    return sign*h*np.array([vphi/y,vrad])


def bilinear(x,y,f,p):
    """
    Computing bilinear interpolation.
    Parameters
    ----------
    x = (x1,x2); y = (y1,y2)
    f = (f11,f12,f21,f22)
    p = (x,y)
    where x,y is the interpolated point and
    fij is the value of the function at the
    point (xi,yj).
    Output
    ------
    f(p): Float.
          The interpolated value of the function f(p) = f(x,y)
    """

    xp  = p[0]; yp   = p[1]; x1  = x[0]; x2  = x[1]
    y1  = y[0]; y2  = y[1];  f11 = f[0]; f12 = f[1]
    f21 = f[2]; f22 = f[3]
    t = (xp-x1)/(x2-x1);    u = (yp-y1)/(y2-y1)

    return (1.0-t)*(1.0-u)*f11 + t*(1.0-u)*f12 + t*u*f22 + u*(1-t)*f21

def get_v(v, x, y):
    """
    For a real set of coordinates (x,y), returns the bilinear            
    interpolated value of a Field class.
    """

    i = int((x-v.xmin)/(v.xmax-v.xmin)*v.nx)
    j = int((y-v.ymin)/(v.ymax-v.ymin)*v.ny)

    if i<0 or j<0 or i>v.data.shape[1]-2 or j>v.data.shape[0]-2:
        return None

    f11 = v.data[j,i]
    f12 = v.data[j,i+1]
    f21 = v.data[j+1,i]
    f22 = v.data[j+1,i+1]
    try:
        x1  = v.x[i]
        x2  = v.x[i+1]
        y1  = v.y[j]
        y2  = v.y[j+1]
        return bilinear((x1,x2),(y1,y2),(f11,f12,f21,f22),(x,y))
    except IndexError:
        return None
    
    
    
def get_stream(vx, vy, x0, y0, nmax=1000000, maxlength=4*np.pi, bidirectional=True, reverse=False):
    """
    Function for computing a streamline.
    Parameters:
    -----------

    x0,y0: Floats.
          Initial position for the stream
    nmax: Integer.
          Maxium number of iterations for the stream.
    maxlength: Float
               Maxium allowed length for a stream
    bidirectional=True
                  If it's True, the stream will be forward and backward computed.
    reverse=False
            The sign of the stream. You can change it mannualy for a single stream,
            but in practice, it's recommeneded to use this function without set reverse
            and setting bidirectional = True.

    Output:
    -------

    If bidirectional is False, the function returns a single array, containing the streamline:
    The format is:

                                      np.array([[x],[y]])

    If bidirectional is True, the function returns a tuple of two arrays, each one with the same
    format as bidirectional=False.
    The format in this case is:

                            (np.array([[x],[y]]),np.array([[x],[y]]))

    This format is a little bit more complicated, and the best way to manipulate it is with iterators.
    For example, if you want to plot the streams computed with bidirectional=True, you should write
    some similar to:

    stream = get_stream(x0,y0)
    ax.plot(stream[0][0],stream[0][1]) #Forward
    ax.plot(stream[1][0],stream[1][1]) #Backward

    """
    
    if bidirectional:
        s0 = get_stream(vx, vy, x0, y0, reverse=False, bidirectional=False, nmax=nmax,maxlength=maxlength)
        s1 = get_stream(vx, vy, x0, y0, reverse=True,  bidirectional=False, nmax=nmax,maxlength=maxlength)
        return (s0,s1)

    l = 0
    x = [x0]
    y = [y0]

    for i in xrange(nmax):
        ds = euler(vx, vy, x0, y0, reverse=reverse)
        if ds[0] == None:
            if(len(x)==1):
                print "There was an error getting the stream, ds was NULL (see get_stream)."
            break
        l += np.sqrt(ds[0]**2+ds[1]**2)
        dx = ds[0]
        dy = ds[1]
        if(np.sqrt(dx**2+dy**2)<1e-13):
            print "Warning (get_stream): ds is very small, maybe you're in a stagnation point."
            print "Try selecting another initial point."
            break
        if l > maxlength:
            print "maxlength reached: ", l
            break
        x0 += dx
        y0 += dy
        x.append(x0)
        y.append(y0)
    return np.array([x,y])

def get_random_streams(vx, vy, xmin=None, xmax=None, ymin=None, ymax=None, n=30, nmax=100000):
    if xmin == None:
        xmin = vx.xmin
    if ymin == None:
        ymin = vx.ymin
    if xmax == None:
        xmax = vx.xmax
    if ymax == None:
        ymax = vx.ymax
    X = xmin + np.random.rand(n)*(xmax-xmin)
    Y = ymin + np.random.rand(n)*(ymax-ymin)
    streams = []
    counter = 0
    for x,y in zip(X,Y):
        stream = get_stream(vx, vy, x, y, nmax=nmax, bidirectional=True)
        streams.append(stream)
        counter += 1
        print "stream ",counter, "done"
    return streams

def plot_random_streams(streams, cartesian=False, **kargs):
    ax = plt.gca()
    print np.shape(streams)
    for stream in streams:
        for sub_stream in stream:
            if cartesian:
                ax.plot(sub_stream[1]*cos(sub_stream[0]),sub_stream[1]*sin(sub_stream[0]),**kargs)
            else:
                ax.plot(sub_stream[0],sub_stream[1],**kargs)
