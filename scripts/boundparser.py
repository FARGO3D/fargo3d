import re
import copy
import sys
import os

STDDIR = "../std/" #Where is "boundary_template.c"

def pformat_read(name,dataformat):
    """
    p-format reader. "Case insensitive version"

    Input:
    ------
    
    name: string
          data filename

    dataformat: string
                regular expresion for the data.
                
    Output:
    -------
    A list of field dictionaries. The main key is the name,
    the rest is a collection of parameters
    """

    datafile = open(name,'r')
    lines = datafile.readlines()
    datafile.close()
    temp_field = ''
    field_name = ''
    field = {}
    fields = []
    init = 0
    for line in lines:
        search = None
        skip = re.match("\s*#+",line) #Very compact comment line!
        if skip != None:
            continue
        search1 = re.match("\s*(\w+)\s*:\s*\n",line)    #without comment
        search2 = re.match("\s*(\w+)\s*:\s*#.*\n",line) #with a comment
        if search1 != None:
            search = search1
        if search2 != None:
            search = search2
        if search != None:
            field_name = search.group(1).lower()
            temp_field = field_name
            if init != 0:
                fields.append(field)
                field = {}
            init = 1
            continue
        if temp_field == field_name:
            field['name'] = field_name #ugly but works! Can be improved!
            search = re.match("\s*(\w+)\s*:\s*("+dataformat+")",line)
            if search != None:
                prop  = search.group(1).lower()
                value = search.group(2).lower()
                field[prop] = value
                continue
    if len(field) != 0:
        fields.append(field)
    return fields

class Library():
    """
    Centering and boundaries library.
    """
    def __init__(self,boundaries, centering):
        self.boundaries = pformat_read(boundaries,"\|.*\|")
        self.centering  = pformat_read(centering,"\w+")
        
class Field():
    def __init__(self,name,library):
        self.library = library
        self.name = name.lower()
        self.centered = self.determine_centered()

    def determine_centered(self):
        for field in self.library.centering:
            if field['name'] == self.name:
                centered = field['staggering']
                break
        return centered

class Setup():
    def __init__(self,setup,boundaries,centering):
        self.fields = pformat_read(setup,"\w+")
        self.library = Library(boundaries,centering)
        fields = []
        for field in self.fields:
            f = Field(field['name'],self.library)
            f.field = field['name'].capitalize()
            f.variable = field['name']
            f.boundaries = {}
            try:
                f.boundaries['ymin'] = field['ymin']
                f.boundaries['ymax'] = field['ymax']
            except:
                pass
            try:
                f.boundaries['zmin'] = field['zmin']
                f.boundaries['zmax'] = field['zmax']
            except:
                pass
                #print "Skipping Z direction..."
            fields.append(f)
        self.fields = fields

class Template():
    def __init__(self,name=STDDIR+"boundary_template.c"):
        self.template = self.read_template(name)
        self.stones = self.get_stones() #Dictionary

    def read_template(self,name):
        template_file = open(name,"r")
        template = template_file.readlines()
        template_file.close()
        return template

    def get_stones(self):
        n = 0
        stones = {}
        for line in self.template:
            stone = re.match(".*(%\w+)\s*",line)
            if stone != None:
                stones[stone.group(1)] = n
            n += 1
        return stones

class Boundary():
    """
    Main class of boundparser.py.
    It is the core of the data. The programmer does not need
    to go to another place, everything is managed from here...
    """
    def __init__(self,side,Setup,Template):
        self.name = side+"_bound.c"
        self.side = side
        self.template = Template
        self.setup = Setup
        self.stones = self.template.stones

        self.process_side()
        self.process_ifields()
        self.process_ofields()
        self.process_internal()
        self.process_external()
        self.process_boundaries()
        self.process_indices()

        self.write_output()

    def write_output(self):
        output = open(self.side+"_bound.c","w")
        for line in self.template.template:
            output.write(line)

    def process_side(self):
        string = "%side"
        n = self.stones[string]
        self.template.template[n] = \
            self.template.template[n].replace(string,self.side+"_cpu")

    def process_ifields(self):
        ifields_lines = ""
        string = "%ifields"
        for field in self.setup.fields:
            if self.side == 'ymin' : direction = "LEFT"
            if self.side == 'ymax' : direction = "RIGHT"
            if self.side == 'zmin' : direction = "DOWN"
            if self.side == 'zmax' : direction = "UP"
            ifields_lines += "INPUT(" + field.field + ");\n"
        n = self.stones[string]
        self.template.template[n] = \
            self.template.template[n] = ifields_lines

    def process_ofields(self):
        ofields_lines = ""
        string = "%ofields"
        for field in self.setup.fields:
            if self.side == 'ymin' : direction = "LEFT"
            if self.side == 'ymax' : direction = "RIGHT"
            if self.side == 'zmin' : direction = "DOWN"
            if self.side == 'zmax' : direction = "UP"
            ofields_lines += "OUTPUT(" + field.field + ");\n"
        n = self.stones[string]
        self.template.template[n] = \
            self.template.template[n] = ofields_lines

    def process_internal(self):
        internal_lines = ""
        string = "%internal"
        n = self.stones[string]
        internal_lines = "  int lgh;\n"+ \
            "  int lghs;\n" + \
            "  int lact;\n" + \
            "  int lacts;\n" + \
            "  int lacts_null;\n"
        self.template.template[n] = internal_lines
        
    def process_external(self):
        pointerfield_lines = ""
        string = "%pointerfield"

        for field in self.setup.fields:
            pointerfield_lines += "  real* "+ \
                field.name + \
                " = "+field.field + \
                "->field_cpu;\n"

        n = self.stones[string]
        self.template.template[n] = \
            self.template.template[n] = pointerfield_lines
        
        string = "%size_y"
        n = self.stones[string]
        if self.side[0] == "y":
            self.template.template[n] = \
                self.template.template[n].replace(string,"NGHY")
        else:
            self.template.template[n] = \
                self.template.template[n].replace(string,"Ny+2*NGHY")

        string = "%size_z"
        n = self.stones[string]
        if self.side[0] == "z":
            self.template.template[n] = \
                self.template.template[n].replace(string,"NGHZ")
        else:
            self.template.template[n] = \
                self.template.template[n].replace(string,"Nz+2*NGHZ")

    def process_boundaries(self):
        boundaries_lines = ""
        global_variables = []
        for field in self.setup.fields:
            if field.centered[-1] == self.side[0]:
                if self.side[:] == "ymax":
                    left_hand = "\t" + "if (j<size_y-1)\n"
                    left_hand += "\t\t" + field.variable + "[lghs] = "
                elif self.side[:] == "zmax":
                    left_hand = "\t" + "if (k<size_z-1)\n"
                    left_hand += "\t\t" + field.variable + "[lghs] = "
                else:
                    left_hand = "\t" + field.variable + "[lghs] = "
            else:
                left_hand = "\t" + field.variable + "[lgh] = "
            right_hand = self.parsing_boundary(field)

#Parsing global variables
            gvariables = re.findall("'[a-z0-9]+'",right_hand)
            for variable in gvariables:
                global_variables.append(variable.replace("'",""))

                
            right_hand = right_hand.replace("'","")
            boundaries_lines += left_hand + right_hand
        
        string = "%boundaries"
        n = self.stones[string]
        self.template.template[n] = boundaries_lines

#Parsing global variables
        string = "%global"
        n = self.stones[string]
        global_variables = set(global_variables)
        variables = ""
        for variable in global_variables:
            variables += "  real " + variable + " = " + variable.upper() + ";\n"
        self.template.template[n] = variables
        

    def parsing_boundary(self,field):
        output = ""
        boundaries = self.setup.library.boundaries
        bound = False
        for boundary in boundaries:
            if boundary['name'] == field.boundaries[self.side]:
                bound = True
                if re.search(self.side[0],field.centered) != None:
                    """
                    Boundaries of the form |a|x|a|
                    """
                    try:
                        bound = boundary['staggered']
                    except KeyError:
                        print "Be careful!", field.name.upper(), "is staggered in", field.centered[-1],\
                            "but you are trying to use a centered condition for it. Please review the "\
                            "definition of", boundary['name'].upper(), "if you want to use this condition."
                        exit()                        
                    groups = re.match("\|(.*)\|(.*)\|(\w*)\|",bound)
                    active = groups.group(1).replace(groups.group(3),field.variable+"[lacts]")
                    ghosts = groups.group(2).replace(groups.group(3),field.variable+"[lacts]")
                    output +=  active + ";\n"
                    if ((field.variable[0] != 'b') or (field.variable[1] != self.side[0])):
                        output += "\t" + field.variable + "[lacts_null] = " + ghosts + ";\n"
                else:
                    """
                    Boundaries of the form |a|a|
                    """
                    try:
                        bound = boundary['centered']
                    except KeyError:
                        print "Be careful!", field.name.upper(), "is centered in", field.centered[-1],\
                            "but you are trying to use a staggered condition for it. Please review the "\
                            "definition of", boundary['name'].upper(), "if you want to use this condition."
                        exit()
                    groups = re.match("\|(.*)\|(\w*)\|",bound)
                    active = groups.group(1).replace(groups.group(2),field.variable+"[lact]")  
                    output += active + ";\n"
        if bound == False:
            print field.boundaries[self.side].upper(),"boundary for",\
                field.name.upper(), "applied on",self.side.upper(), \
                "does not exist! Please, verify if it is defined in boundaries.txt"
            exit()
        return output

    def process_indices(self):
        indices_lines = ""
        if self.side == 'ymin':
            indices_lines += "\n\t" +"lgh = l;\n"            
            indices_lines += "\t" +"lghs = l;\n"
            indices_lines += "\t" +"lact = i + (2*nghy-j-1)*pitch + k*stride;\n"
            indices_lines += "\t"+"lacts = i + (2*nghy-j)*pitch + k*stride;\n"
            indices_lines += "\t"+"lacts_null = i + nghy*pitch + k*stride;\n"
            indices_lines += "\t" +"jgh = j;\n"
            indices_lines += "\t" +"jact = (2*nghy-j-1);\n\n"
        if self.side == 'ymax':
            indices_lines += "\n\t" +"lgh = i + (ny+nghy+j)*pitch + k*stride;\n"
            indices_lines += "\t" +"lghs = i + (ny+nghy+1+j)*pitch + k*stride;\n"
            indices_lines += "\t" +"lact = i + (ny+nghy-1-j)*pitch + k*stride;\n"
            indices_lines += "\t" +"lacts = i + (ny+nghy-1-j)*pitch + k*stride;\n"
            indices_lines += "\t" +"lacts_null = i + (ny+nghy)*pitch + k*stride;\n"
            indices_lines += "\t" +"jgh = (ny+nghy+j);\n"
            indices_lines += "\t" +"jact = (ny+nghy-1-j);\n\n"
        if self.side == 'zmin':
            indices_lines += "\n\t" +"lgh = l;\n"            
            indices_lines += "\t" +"lghs = l;\n"
            indices_lines += "\t" +"lact = i + j*pitch + (2*nghz-k-1)*stride;\n"
            indices_lines += "\t"+"lacts = i + j*pitch + (2*nghz-k)*stride;\n"
            indices_lines += "\t"+"lacts_null = i + j*pitch + nghz*stride;\n"
            indices_lines += "\t" +"kgh = k;\n"
            indices_lines += "\t" +"kact = (2*nghz-k-1);\n\n"
        if self.side == 'zmax':
            indices_lines += "\n\t" +"lgh = i + j*pitch + (nz+nghz+k)*stride;\n"
            indices_lines += "\t" +"lghs = i + j*pitch + (nz+nghz+1+k)*stride;\n"
            indices_lines += "\t" +"lact = i + j*pitch + (nz+nghz-1-k)*stride;\n"
            indices_lines += "\t" +"lacts = i + j*pitch + (nz+nghz-1-k)*stride;\n"
            indices_lines += "\t" +"lacts_null = i + j*pitch + (nz+nghz)*stride;\n"
            indices_lines += "\t" +"kgh = (nz+nghz+k);\n"
            indices_lines += "\t" +"kact = (nz+nghz-1-k);\n\n"
        string = "%boundaries"
        n = self.stones[string]
        self.template.template[n] = indices_lines + self.template.template[n]

def write_mute(side, template):

    template1 = copy.deepcopy(template)
    template2 = copy.deepcopy(template)
    stones = template.stones

    output1 = open(side+"min_bound.c","w")
    template1.template[stones['%side']] = \
        template1.template[stones['%side']].replace("%side",side+"min_cpu")
    for key in stones.keys():
        if key == "%size_y":
            template1.template[stones['%size_y']] = \
                template1.template[stones['%size_y']].replace("%size_y","1")
            continue
        if key == "%size_z":
            template1.template[stones['%size_z']] = \
                template1.template[stones['%size_z']].replace("%size_z","1")
            continue
        if key != "%side":
            n = template1.stones[key]
            template1.template[n] = ""
    for line in template1.template:
        output1.write(line)
    output1.close()

    output2 = open(side+"max_bound.c","w")
    template2.template[stones['%side']] = \
        template2.template[stones['%side']].replace("%side",side+"max_cpu")
    for key in stones.keys():
        if key == "%size_y":
            template2.template[stones['%size_y']] = \
                template2.template[stones['%size_y']].replace("%size_y","1")
            continue
        if key == "%size_z":
            template2.template[stones['%size_z']] = \
                template2.template[stones['%size_z']].replace("%size_z","1")
            continue
        if key != "%side":
            n = template2.stones[key]
            template2.template[n] = ""
            continue
    for line in template2.template:
        output2.write(line)
    output2.close()

def process_arguments(arguments):
    SETUP = BOUNDARIES = CENTERING = None
    for argument in arguments:
        search = re.search(".*\.bound",argument)
        if search!=None:
            SETUP = search.group(0)
        search = re.search(".*boundaries.txt",argument)
        if search!=None:
            BOUNDARIES = search.group(0)
        search = re.search(".*centering.txt",argument)
        if search!=None:
            CENTERING = search.group(0)
    return SETUP, BOUNDARIES, CENTERING

if __name__ == '__main__':

    SETUP, BOUNDARIES, CENTERING = process_arguments(sys.argv[1:])
    if SETUP == None:
        print "Check your setup.bound file..."
    if BOUNDARIES == None:
        print "Check your boundaries.txt file..."
    if SETUP == None:
        print "Check your centering.txt file..."
    if SETUP == None or BOUNDARIES == None or CENTERING == None:
        exit()

    template = Template()

    try:
        setup = Setup(SETUP, BOUNDARIES, CENTERING)
    except:
        write_mute("y",copy.deepcopy(template))
        write_mute("z",copy.deepcopy(template))
        print "================================="
        print "Warning: Some file is missing..." 
        "\nCheck the files: \n" + SETUP + "\n" \
            + BOUNDARIES + "\n" + CENTERING + \
            "\nThis version was compiled with"
        "periodic boundaries."
        print "================================="
        exit()

    try:
        left  = Boundary("ymin",
                         copy.deepcopy(setup),
                         copy.deepcopy(template))
        right = Boundary("ymax", 
                         copy.deepcopy(setup), 
                         copy.deepcopy(template))
    except KeyError:
        write_mute("y", copy.deepcopy(template))
        print "Skipping boundaries in Y. Not defined."

    try:
        down  = Boundary("zmin", 
                         copy.deepcopy(setup),
                         copy.deepcopy(template))
        up    = Boundary("zmax", 
                         copy.deepcopy(setup), 
                         copy.deepcopy(template))
    except KeyError:
        write_mute("z", copy.deepcopy(template))
        print "Skipping boundaries in Z. Not defined."
