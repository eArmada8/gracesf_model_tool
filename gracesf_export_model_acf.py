# Tool to export model data from the model .acf files used by Tales of Graces f (PS3).
# Thank you to Mc-muffin (github.com/Mc-muffin) for the decompression code.
#
# Usage:  Run by itself without commandline arguments and it will search for .acf model archives
# and export a .gib file.
#
# For command line options, run:
# /path/to/python3 gracesf_export_model_acf.py --help
#
# Requires gracesf_export_model.py and lib_fmtibvb.py, put both in the same directory
#
# GitHub eArmada8/gracesf_model_tool

try:
    import struct, json, numpy, copy, glob, os, sys
    from typing import Optional
    from gracesf_export_model import *
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

DICT_SIZE = 0x1000  # 4096

# The entire CompToE implementation is written by Ethanol and is here with permission, thank you!
def init_dictionary() -> bytearray:
    # Init LZSS dictionary, in Okumura's version is all 0
    tbl = bytearray(DICT_SIZE)
    idx = 0
    for v in range(256):
        if idx + 8 > DICT_SIZE:
            break
        tbl[idx + 0] = v
        tbl[idx + 1] = 0
        tbl[idx + 2] = v
        tbl[idx + 3] = 0
        tbl[idx + 4] = v
        tbl[idx + 5] = 0
        tbl[idx + 6] = v
        tbl[idx + 7] = 0
        idx += 8
    for v in range(256):
        if idx + 7 > DICT_SIZE:
            break
        tbl[idx + 0] = v
        tbl[idx + 1] = 0xFF
        tbl[idx + 2] = v
        tbl[idx + 3] = 0xFF
        tbl[idx + 4] = v
        tbl[idx + 5] = 0xFF
        tbl[idx + 6] = v
        idx += 7
    return tbl

def decode(compressed: bytes, dict_buf: Optional[bytearray] = None) -> bytes:
    if dict_buf is None:
        dict_buf = init_dictionary()
    else:
        if len(dict_buf) < DICT_SIZE:
            raise ValueError("dict_buf must be at least 0x1000 bytes")
    out = bytearray()
    ip = 0
    ip_end = len(compressed)
    sp20 = 0x12
    sp1C = (0x1000 - sp20)  # initial write pointer in dict
    flags = 0 
    while ip < ip_end:
        flags >>= 1
        if not (flags & 0x100):
            flags = compressed[ip] | 0xFF00
            ip += 1
        if flags & 1:
            # literal
            b = compressed[ip]
            out.append(b)
            dict_buf[sp1C] = b
            sp1C = (sp1C + 1) & 0xFFF
            ip += 1
        else:
            # copy from dictionary: read two bytes
            sp12 = compressed[ip + 0]
            sp14 = compressed[ip + 1]
            ip += 2
            sp12 |= ((sp14 & 0xF0) << 4)
            length = (sp14 & 0xF) + 2
            i = 0
            # copy length bytes from (sp12 + offset) & 0xFFF
            while length >= i:
                b = dict_buf[(sp12 + i) & 0xFFF]
                out.append(b)
                dict_buf[sp1C] = b
                sp1C = (sp1C + 1) & 0xFFF
                i += 1
    return bytes(out)

def decode_run(compressed: bytes, dict_buf: Optional[bytearray] = None) -> bytes:
    if dict_buf is None:
        dict_buf = init_dictionary()
    else:
        if len(dict_buf) < DICT_SIZE:
            raise ValueError("dict_buf must be at least 0x1000 bytes")
    out = bytearray()
    ip = 0
    ip_end = len(compressed)
    sp20 = 0x11
    sp1C = (0x1000 - sp20)  # initial write pointer in dict
    flags = 0
    while ip < ip_end:
        flags >>= 1
        if not (flags & 0x100):
            flags = compressed[ip] | 0xFF00
            ip += 1
        if flags & 1:
            # literal
            b = compressed[ip]
            ip += 1
            out.append(b)
            dict_buf[sp1C] = b
            sp1C = (sp1C + 1) & 0xFFF
        else:
            # read two bytes
            sp12 = compressed[ip + 0]
            sp14 = compressed[ip + 1]
            ip += 2
            sp12 |= ((sp14 & 0xF0) << 4)
            length = (sp14 & 0xF) + 2
            if length < 0x11:
                # simple copy from dictionary
                for offset in range(length+1):
                    src_idx = (sp12 + offset) & 0xFFF
                    b = dict_buf[src_idx]
                    out.append(b)
                    dict_buf[sp1C] = b
                    sp1C = (sp1C + 1) & 0xFFF
            else:
                # extended cases
                if sp12 < 0x100:
                    fill_byte = compressed[ip]
                    ip += 1
                    rep_len = sp12 + 0x12
                else:
                    fill_byte = sp12 & 0xFF
                    rep_len = ((sp12 >> 8) + 2)
                for _ in range(rep_len+1):
                    out.append(fill_byte)
                    dict_buf[sp1C] = fill_byte
                    sp1C = (sp1C + 1) & 0xFFF
    return bytes(out)

def sc_decode(data: bytes, dict_buf: Optional[bytearray] = None) -> bytes:
    if len(data) < 9:
        raise ValueError("data too short for header (need at least 9 bytes)")
    type = data[0]
    comp_size = int.from_bytes(data[1:5], "little")
    payload_start = 9
    end = payload_start + comp_size
    comp = data[payload_start:end]
    if type == 0:
        # no compression
        return bytes(comp)
    elif type == 1:
        # Okumura's LZSS
        return decode(comp, dict_buf)
    elif type == 3:
        # LZSS + RLE
        return decode_run(comp, dict_buf)
    else:
        raise ValueError(f"Invalid decode type {type}")

def process_tex_f (f, base_name):
    print("Processing {}...".format(base_name))
    set_endianness('>')
    magic = f.read(4)
    if magic == b'FPS4':
        header = struct.unpack(">3I2H2I", f.read(24))
        assert header[0] == 3
        toc = []
        for i in range(header[0]):
            toc.append(struct.unpack(">3I".format(e), f.read(12))) # offset, padded length, true length
        textures = read_texture_section (f, toc[0][0], toc[1][0], toc[1][2])
        if not os.path.exists('textures'):
            os.mkdir('textures')
        for i in range(len(textures)):
            open('textures/' + textures[i]['name'], 'wb').write(textures[i]['data'])
    return True

def process_acf (acf_file, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    print("Processing {}...".format(acf_file))
    set_endianness('>')
    with open(acf_file, 'rb') as f:
        magic = f.read(4)
        if magic == b'FPS4':
            header = struct.unpack(">3I2H2I", f.read(24)) # num_entries, unk, header_size, entry_len, bitmask, unk*2
            toc = []
            for i in range(header[0]):
                start = f.tell()
                toc_entry = struct.unpack(">3i".format(e), f.read(12)) # offset, padded length, true length
                toc_name = read_string(f, f.tell())
                f.seek(start + (header[3]))
                toc.append({'name': toc_name, 'offset': toc_entry[0],
                    'padded_size': toc_entry[2], 'true_size': toc_entry[2]})
            for i in range(len(toc)):
                if toc[i]['true_size'] > 0:
                    if 'MDL' in toc[i]['name'] or 'TEX' in toc[i]['name']:
                        f.seek(toc[i]['offset'])
                        unc_data = sc_decode(f.read(toc[i]['true_size']))
                    if 'MDL' in toc[i]['name']:
                        with io.BytesIO(unc_data) as ff:
                            process_model(ff, toc[i]['name'][:-4], overwrite = overwrite,
                                write_raw_buffers = write_raw_buffers, write_binary_gltf = write_binary_gltf)
                    elif 'TEX' in toc[i]['name']:
                        with io.BytesIO(unc_data) as ff:
                            process_tex_f(ff, toc[i]['name'][:-4])

if __name__ == "__main__":
    # Set current directory
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--textformat', help="Write gltf instead of glb", action="store_false")
        parser.add_argument('-d', '--dumprawbuffers', help="Write fmt/ib/vb/vgmap files in addition to glb", action="store_true")
        parser.add_argument('-o', '--overwrite', help="Overwrite existing files", action="store_true")
        parser.add_argument('acf_file', help="Name of model acf file to process.")
        args = parser.parse_args()
        if os.path.exists(args.acf_file) and args.acf_file[-4:].lower() in ['.acf', '.dat']:
            process_acf(args.acf_file, overwrite = args.overwrite, \
                write_raw_buffers = args.dumprawbuffers, write_binary_gltf = args.textformat)
    else:
        acf_files = glob.glob('*.acf') + glob.glob('*.dat')
        for acf_file in acf_files:
            process_acf(acf_file)