import re
import os
import argparse
import textwrap

parser = argparse.ArgumentParser(prog="boundparser.py",
    description="Compile .bound files to corresponding sources",
    epilog="This script is part of FARGO3D.")
parser.add_argument("inputs", nargs="+")
parser.add_argument("-t", "--template", default="boundary_template.c")
parser.add_argument("-b", "--boundaries", default="boundaries.txt")
parser.add_argument("-c", "--centering", default="centering.txt")
parser.add_argument("-o", "--outdir", default="")
args = parser.parse_args()

def pformat_read(datafile, dataformat):
    """Case-insensitive p-format reader.

    datafile -- name of a file to open
    dataformat -- regex for the data to extract

    Returns a dictionary of dictionaries.
    """

    fields = dict()

    # These nasty-looking regex patterns will match an entire
    # file entry (pat1) and each sub-entry thereof (pat2). Comments
    # will be skipped naturally because they won't match. This is
    # more succinct than reading line-by-line, at the expense of being
    # a little terrifying to read. For future reference:
    #   [^\S\r\n] matches non-newline whitespace
    #   (?:#.*)? will match without capturing a #-leading comment or nothing
    pat2 = r'[^\S\r\n]*(\w+)[^\S\r\n]*:[^\S\r\n]*(' + dataformat + r')[^\S\r\n]*(?:#.*)?'
    pat1 = r'[^\S\r\n]*(\w+)[^\S\r\n]*:[^\S\r\n]*(?:#.*)?((?:\n' + pat2 + r')+)'

    rgx1 = re.compile(pat1)
    rgx2 = re.compile(pat2)

    with open(datafile, 'r') as dat:
        alldat = dat.read()

    for m1 in rgx1.finditer(alldat):
        field_name = m1.group(1).lower()
        field_props = dict()

        for m2 in rgx2.finditer(m1.group(2)):
            prop_name  = m2.group(1).lower()
            prop_value = m2.group(2).lower()
            field_props[prop_name] = prop_value

        fields[field_name] = field_props

    return fields

def get_indices(side):
    indices_lines = ""
    if side == 'ymin':
        indices_lines = """\
            lgh = l;
            lghs = l;
            lact = i + (2*nghy-j-1)*pitch + k*stride;
            lacts = i + (2*nghy-j)*pitch + k*stride;
            lacts_null = i + nghy*pitch + k*stride;
            lacts_null_mirror = i + (nghy+1)*pitch + k*stride;
            jgh = j;
            jact = (2*nghy-j-1);
            """
    if side == 'ymax':
        indices_lines = """\
            lgh = i + (ny+nghy+j)*pitch + k*stride;
            lghs = i + (ny+nghy+1+j)*pitch + k*stride;
            lact = i + (ny+nghy-1-j)*pitch + k*stride;
            lacts = i + (ny+nghy-1-j)*pitch + k*stride;
            lacts_null = i + (ny+nghy)*pitch + k*stride;
            lacts_null_mirror = i + (ny+nghy-1)*pitch + k*stride;
            jgh = (ny+nghy+j);
            jact = (ny+nghy-1-j);
            """
    if side == 'zmin':
        indices_lines = """\
            lgh = l;
            lghs = l;
            lact = i + j*pitch + (2*nghz-k-1)*stride;
            lacts = i + j*pitch + (2*nghz-k)*stride;
            lacts_null = i + j*pitch + nghz*stride;
            lacts_null_mirror = i + j*pitch + (nghz+1)*stride;
            kgh = k;
            kact = (2*nghz-k-1);
            """
    if side == 'zmax':
        indices_lines = """\
            lgh = i + j*pitch + (nz+nghz+k)*stride;
            lghs = i + j*pitch + (nz+nghz+1+k)*stride;
            lact = i + j*pitch + (nz+nghz-1-k)*stride;
            lacts = i + j*pitch + (nz+nghz-1-k)*stride;
            lacts_null = i + j*pitch + (nz+nghz)*stride;
            lacts_null_mirror = i + j*pitch + (nz+nghz-1)*stride;
            kgh = (nz+nghz+k);
            kact = (nz+nghz-1-k);
            """
    return textwrap.dedent(indices_lines)

def parse_boundary(side, name, field, boundary, stagger):
    output = ""
    bound = False

    bnd = boundary[field[side]]
    stag = centering[name]['staggering']

    if side[0] in stag:
        # boundaries of form |a|x|a|
        try:
            bound = bnd['staggered']
        except KeyError:
            fieldname = name.upper()
            bndname = field[side].upper()
            msg = f"{fieldname} is staggered in {stag} but you are trying " \
                "to use a centered condition for it. Please review the " \
                "definition of {bndname} if you want to use this condition."
            raise ValueError(msg)

        m = re.match(r"\|(.*)\|(.*)\|(\w*)\|", bound)
        g1, g2, g3 = m.group(1, 2, 3)
        active = g1.replace(g3, name + "[lacts]")
        ghosts = g2.replace(g3, name + "[lacts]")
        ghostsm = g2.replace(g3, name + "[lacts_null_mirror]")
        output +=  f"{active};"
        if (name[0] != 'b') or (name[1] != side[0]):
            output += f"\n{name}[lacts_null] = {ghostsm};"
    else:
        # boundaries of form |a|a|
        try:
            bound = bnd['centered']
        except KeyError:
            fieldname = name.upper()
            bndname = field[side].upper()
            msg = f"{fieldname} is centered in {stag} but you are trying " \
                "to use a staggered condition for it. Please review the " \
                "definition of {bndname} if you want to use this condition."
            raise ValueError(msg)

        m = re.match(r"\|(.*)\|(\w*)\|", bound)
        g1, g2 = m.group(1, 2)
        active = g1.replace(g2, name + "[lact]")
        output += f"{active};"

    if not bound:
        fieldname = name.upper()
        bndname = field[side].upper()
        sidename = side.upper()
        msg = f"{bndname} boundary for {fieldname} applied on {sidename} " \
            "does not exist! Verify that it is defined in the boundaries file."
        raise ValueError(msg)

    return output

def format_template(template, number, side, fields, boundaries, centering):
    globs = set()

    def gen_substitutions():
        nonlocal globs

        for name, field in fields.items():
            namecap = name.capitalize()
            ifield = f'INPUT({namecap});'
            ofield = f'OUTPUT({namecap});'
            pfield = f'real *{name} = {namecap}->field_cpu;'

            bnd = boundaries[field[side]]
            stag = centering[name]['staggering']

            rhs = parse_boundary(side, name, field, boundaries, centering)

            # any variables appearing in rhs should be added to the growing set
            for m in re.finditer(r"'([a-z0-9]+)'", rhs):
                globs.add(m.group(1))
            rhs = rhs.replace("'", "")

            if stag == side[0]:
                # field is face-centered in this direction
                if side == 'ymax':
                    expr = f'''\
                        if (j<size_y-1) {{
                          {name}[lghs] = {rhs}
                        }}
                        '''
                elif side == 'zmax':
                    expr = f'''\
                        if (k<size_z-1) {{
                          {name}[lghs] = {rhs}
                        }}
                        '''
                else:
                    expr = f'{name}[lghs] = {rhs}'
            else:
                # field is cell-centered in this direction
                expr = f'{name}[lgh] = {rhs}'

            ifield = textwrap.indent(textwrap.dedent(ifield), " " * 2)
            ofield = textwrap.indent(textwrap.dedent(ofield), " " * 2)
            pfield = textwrap.indent(textwrap.dedent(pfield), " " * 2)

            inds = get_indices(side)
            expr = '\n'.join([inds, textwrap.dedent(expr)])
            expr = textwrap.indent(expr, " " * 8)

            yield ifield, ofield, pfield, expr

    ifields, ofields, pfields, exprs = list(zip(*list(gen_substitutions())))

    ifields = '\n'.join(ifields)
    ofields = '\n'.join(ofields)
    pfields = '\n'.join(pfields)
    exprs = '\n'.join(exprs)

    def gen_globals(globs):
        for name in globs:
            uppername = name.upper()
            yield f'real {name} = {uppername};'

    globs = textwrap.indent('\n'.join(list(gen_globals(globs))), " " * 2)

    internal = """\
        int lgh;
        int lghs;
        int lact;
        int lacts;
        int lacts_null_mirror;
        int lacts_null;
        """
    internal = textwrap.indent(textwrap.dedent(internal), "  ")

    size_y = 'NGHY' if side[0] == 'y' else 'Ny+2*NGHY'
    size_z = 'NGHZ' if side[0] == 'z' else 'Nz+2*NGHZ'

    # for the function name only, need a more explicit descriptor
    sideex = f'{side}_{number:d}_cpu'

    tfmt = template.format(side=sideex, ifields=ifields, ofields=ofields,
        internal=internal, pointerfield=pfields, size_y=size_y, size_z=size_z,
        globals=globs, boundaries=exprs)

    with open(os.path.join(args.outdir, f'{side}_bound_{number:d}.c'), 'w') as outfile:
        outfile.write(tfmt)

# read the template once
with open(args.template, "r") as tfile:
    template = tfile.read()

# read the boundary and centering files
boundaries = pformat_read(args.boundaries, r"\|.*\|")
centering  = pformat_read(args.centering, r"\w+")

for arg in args.inputs:
    m = re.search(r".*\.bound.(\d+)", arg)
    if m is None:
        raise ValueError("Unexpected filename {}; should be " \
                         "[setup].bound.[number]".format(arg))

    number = int(m.group(1))
    fields = pformat_read(arg, r"\w+")

    try:
        left  = format_template(template, number, "ymin", fields, boundaries, centering)
        right = format_template(template, number, "ymax", fields, boundaries, centering)
    except KeyError:
        pass

    try:
        up    = format_template(template, number, "zmin", fields, boundaries, centering)
        down  = format_template(template, number, "zmax", fields, boundaries, centering)
    except KeyError:
        pass
