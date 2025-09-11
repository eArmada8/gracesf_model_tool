# Tool to export texture data from the extra texture format used by Tales of Graces f (PS3).
#
# Usage:  Run by itself without commandline arguments and it will search for .TEX files
# and export .dds files.
#
# For command line options, run:
# /path/to/python3 gracesf_export_texture.py --help
#
# Requires gracesf_export_model.py and lib_fmtibvb.py, put in the same directory
#
# GitHub eArmada8/gracesf_model_tool

try:
    import struct, json, numpy, copy, glob, os, sys
    from gracesf_export_model import *
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

# Global variable, do not edit
e = '<'

def process_tex (tex_file):
    print("Processing {}...".format(tex_file))
    base_name = tex_file[:-4]
    with open(tex_file, 'rb') as f:
        e = '>'
        set_endianness('>') # Figure out later how to determine this
        magic = f.read(4)
        if magic == b'FPS4':
            header = struct.unpack("{}6I".format(e), f.read(24))
            assert header[0] == 3
            toc = []
            for i in range(header[0]):
                toc.append(struct.unpack("{}3I".format(e), f.read(12))) # offset, padded length, true length
            textures = read_texture_section (f, toc[0][0], toc[1][0], toc[1][2])
            if not os.path.exists('textures'):
                os.mkdir('textures')
            for i in range(len(textures)):
                open('textures/' + textures[i]['name'], 'wb').write(textures[i]['data'])
    return True


if __name__ == "__main__":
    # Set current directory
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('tex_file', help="Name of model file to process.")
        args = parser.parse_args()
        if os.path.exists(args.tex_file) and args.tex_file[-4:].upper() == '.TEX':
            process_tex(args.tex_file)
    else:
        tex_files = glob.glob('*.TEX')
        for tex_file in tex_files:
            process_tex(tex_file)
