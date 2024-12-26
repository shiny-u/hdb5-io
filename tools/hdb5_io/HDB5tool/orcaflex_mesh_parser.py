#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import yaml
import numpy as np
import argparse
import os

try:
    import argcomplete
    acok=True
except:
    acok=False

# -----------------------------------------------------
# Parser
# -----------------------------------------------------

def get_parser(parser):

    parser.add_argument('--input', '-i', action='store', metavar="input", help="Input file")
    parser.add_argument('--output', '-o', action='store', metavar='output', help="Output file prefix")
    parser.add_argument('--structure', '-s', nargs='+', type=str, help="List of structures")

    return parser

def get_by_name(reader, name):

    for data in reader:
        if data["Name"] == name:
            return data

    return None

# -----------------------------------------------------
# Reader/Writer
# -----------------------------------------------------

def write(name, vertices, panels, output):

    outfile = os.path.join(output, "mesh_orca_{}.obj".format(name))
    print("debug : outpath = ", outfile)

    nv = vertices.shape[0]
    nf = panels.shape[0]

    with open(outfile, 'w') as f:

        for i in range(nv):
            vi = vertices[i, :]
            f.write("v {:.5f} {:.5f} {:.5f}\n".format(vi[0], vi[1], vi[2]))

        for i in range(nf):
            fi = panels[i, :]
            f.write("f {} {} {} {}\n".format(fi[0], fi[1], fi[2], fi[3]))

    return

def reader(file, output, structure_names):

    with open(file, 'r') as f:
        reader = yaml.safe_load(f)

    for structure_name in structure_names:
        vessel = get_by_name(reader["Vessels"], structure_name)
        type_name = vessel["VesselType"]
        data = get_by_name(reader["VesselTypes"], type_name)

        vertices = np.array(data["VertexX, VertexY, VertexZ"], dtype=float)
        panels = np.array(data["PanelVertexIndex1, PanelVertexIndex2, PanelVertexIndex3, PanelVertexIndex4"], dtype=int)

        ##CC debug
        print("debug : vertices.shape = ", vertices.shape, " ; panels.shape = ", panels.shape)
        ##CC

        write(structure_name, vertices, panels, output)

    return


# -----------------------------------------------------
# Main executable
# -----------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=""" --- Extract Mesh from Orcaflex YAML --- 
            A python script to extract the mesh data from Orcaflex YAML text data file. 
            \n\n Example of use :
             \n\n orcaflex_mesh_parser --help
             """,
        formatter_class= argparse.RawDescriptionHelpFormatter
    )

    parser = get_parser(parser)

    if acok:
        argcomplete.autocomplete(parser)

    args, unknown = parser.parse_known_args()

    reader(args.input, args.output, args.structure)

    return


if __name__ == "__main__":
    main()