# Tool to export model data from the model format used by Tales of Graces (Wii).
# Texture decoding courtesy of the Dolphin emulator project, much gratitude to them, plus
# to Mc-muffin for help decoding CMPR textures.
#
# Usage:  Run by itself without commandline arguments and it will search for model files
# and export a .gib file.
#
# For command line options, run:
# /path/to/python3 graces_wii_export_model.py --help
#
# Requires gracesf_export_model.py and lib_fmtibvb.py, put both in the same directory
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

def make_fmt_wii (num_uvs, has_weights = True):
    fmt = {'stride': '0', 'topology': 'trianglelist', 'format':\
        "DXGI_FORMAT_R16_UINT", 'elements': []}
    element_id, stride = 0, 0
    semantic_index = {'TEXCOORD': 0} # Counters for multiple indicies
    elements = []
    for i in range(1 + num_uvs + (2 if has_weights else 0)):
            # I think order matters in this dict, so we will define the entire structure with default values
            element = {'id': '{0}'.format(element_id), 'SemanticName': '', 'SemanticIndex': '0',\
                'Format': '', 'InputSlot': '0', 'AlignedByteOffset': str(stride),\
                'InputSlotClass': 'per-vertex', 'InstanceDataStepRate': '0'}
            if i == 0:
                element['SemanticName'] = 'POSITION'
                element['Format'] = 'R32G32B32_FLOAT'
                stride += 12
            elif i == 3+num_uvs-2:
                element['SemanticName'] = 'BLENDWEIGHTS'
                element['Format'] = 'R32G32B32A32_FLOAT'
                stride += 16
            elif i == 3+num_uvs-1:
                element['SemanticName'] = 'BLENDINDICES'
                element['Format'] = 'R8G8B8A8_UINT'
                stride += 4
            else:
                element['SemanticName'] = 'TEXCOORD'
                element['SemanticIndex'] = str(semantic_index['TEXCOORD'])
                element['Format'] = 'R32G32_FLOAT'
                semantic_index['TEXCOORD'] += 1
                stride += 8
            element_id += 1
            elements.append(element)
    fmt['stride'] = str(stride)
    fmt['elements'] = elements
    return(fmt)

def read_mesh_wii (mesh_info, f):
    def read_idx (f, offset, buffer_len):
        f.seek(offset)
        strips = []
        vert_uv_tuples = []
        start = f.read(1)
        while start == b'\x98' and f.tell() < (offset + buffer_len):
            num, = struct.unpack("{}H".format(e), f.read(2))
            raw_strip = list(struct.unpack("{0}{1}H".format(e, num * 2), f.read(num * 4)))
            vert_idx_strip = [raw_strip[i] for i in range(len(raw_strip)) if i % 2 == 0]
            vert_uv_tuples.extend([(raw_strip[i*2], raw_strip[i*2+1]) for i in range(len(raw_strip)//2)])
            strips.append(vert_idx_strip)
            start = f.read(1)
        fused_strip = strips[0]
        for i in range(1, len(strips)):
            fused_strip.append(-1)
            fused_strip.extend(strips[i])
        return(fused_strip, vert_uv_tuples)
    def read_floats (f, num):
        return(list(struct.unpack("{0}{1}f".format(e, num), f.read(num * 4))))
    def read_bytes (f, num):
        return(list(struct.unpack("{0}{1}B".format(e, num), f.read(num))))
    def read_interleaved_floats (f, num, stride, total):
        vecs = []
        padding = stride - (num * 4)
        for i in range(total):
            vecs.append(read_floats(f, num))
            f.seek(padding, 1)
        return(vecs)
    def read_interleaved_bytes (f, num, stride, total):
        vecs = []
        padding = stride - (num)
        for i in range(total):
            vecs.append(read_bytes(f, num))
            f.seek(padding, 1)
        return(vecs)
    def fix_weights (weights):
        for _ in range(3):
            weights = [x+[round(1-sum(x),6)] if len(x) < 4 else x for x in weights]
        return(weights)
    # Indices
    idx_buffer, vert_uv_tuples = read_idx (f, mesh_info['idx_offset'], mesh_info['total_idx'])
    total_uvs = max([x[1] for x in vert_uv_tuples]) + 1
    calc_total_verts = max([x[0] for x in vert_uv_tuples]) + 1 # 0x400 meshes do not have a reliable counter
    vert_to_uv_dict = {x[0]:x[1] for x in vert_uv_tuples}
    # Vertices
    f.seek(mesh_info['vert_offset'])
    #uv_stride = mesh_info['uv_stride'] # doesn't have the padding
    num_uv_maps = mesh_info['flags2'] & 0xF
    uv_stride = num_uv_maps * 8
    verts = []
    norms = []
    if mesh_info['flags'] & 0xF00 == 0x100:
        blend_idx = []
        weights = []
        num_verts = mesh_info['num_verts']
        current_sum = sum(num_verts)
        total_verts = current_sum
        while not current_sum == 0:
            safety_check = struct.unpack("{}4I".format(e), f.read(16))
            if not safety_check == (0,0,0,0): # Not sure why, sometimes there is extra padding
                f.seek(-16,1)
            start_offset = f.tell()
            for j in range(len(num_verts)):
                vert_offset = f.tell()
                blend_idx_offset = vert_offset + 12
                weights_offset = blend_idx_offset + 4
                stride = 16 + (j * 4)
                end_offset = f.tell() + (num_verts[j] * stride)
                f.seek(vert_offset)
                verts.extend(read_interleaved_floats(f, 3, stride, num_verts[j]))
                f.seek(blend_idx_offset)
                blend_idx.extend(read_interleaved_bytes(f, 4, stride, num_verts[j]))
                if j > 0:
                    f.seek(weights_offset)
                    weights.extend(read_interleaved_floats(f, j, stride, num_verts[j]))
                else:
                    weights.extend([[1.0] for _ in range(num_verts[j])])
                f.seek(end_offset)
            weights = fix_weights(weights)
            if total_verts == mesh_info['total_verts']:
                break
            num_verts = struct.unpack("{}4I".format(e), f.read(16))
            current_sum = sum(num_verts)
            total_verts += current_sum
            while f.tell() % 16:
                f.seek(1,1) # Skip padding
    elif mesh_info['flags'] & 0xF00 == 0x400:
        total_verts = calc_total_verts
        f.seek(mesh_info['uv_offset'] + 0x1c)
        verts.extend([read_floats(f, 3) for _ in range(total_verts)])
    elif mesh_info['flags'] & 0xF00 == 0x700:
        # No weights, so only mesh_info['num_verts'][0] is non-zero
        total_verts = mesh_info['total_verts']
        vert_offset = f.tell()
        verts.extend([read_floats(f, 3) for _ in range(total_verts)])
    uv_maps = []
    if mesh_info['flags'] & 0xF00 in [0x100, 0x700]:
        for i in range(num_uv_maps):
            f.seek(mesh_info['uv_offset'] + 0x18 + (i * 8))
            raw_uv_map = read_interleaved_floats (f, 2, uv_stride, total_uvs)
            uv_maps.append([raw_uv_map[vert_to_uv_dict[i]] for i in range(total_verts)])
    elif mesh_info['flags'] & 0xF00 == 0x400:
        for i in range(num_uv_maps):
            # Do not f.seek() - already at UVs
            raw_uv_map = read_interleaved_floats (f, 2, uv_stride, total_uvs)
            uv_maps.append([raw_uv_map[vert_to_uv_dict[i]] for i in range(total_verts)])
    fmt = make_fmt_wii(len(uv_maps), True)
    vb = [{'Buffer': verts}]
    for uv_map in uv_maps:
        vb.append({'Buffer': uv_map})
    if mesh_info['flags'] & 0xF00 == 0x100:
        vb.append({'Buffer': weights})
        vb.append({'Buffer': blend_idx})
    elif mesh_info['flags'] & 0xF00 in [0x400, 0x700]:
        vb.append({'Buffer': [[1.0, 0.0, 0.0, 0.0] for _ in range(len(verts))]})
        vb.append({'Buffer': [[0, 0, 0, 0] for _ in range(len(verts))]})
    return({'fmt': fmt, 'vb': vb, 'ib': trianglestrip_to_list(idx_buffer)})

def read_mesh_section_wii (f, start_offset, uv_start_offset):
    f.seek(start_offset)
    header = struct.unpack("{}9I".format(e), f.read(36)) #unk0, size, unk1, num_meshes, palette_count, unknown * 4
    num_meshes = header[3]
    palette_count = header[4]
    dat0, num_verts = [],[]
    for _ in range(num_meshes):
        dat0.append(struct.unpack("{}4f".format(e), f.read(16)))
        num_verts.append(struct.unpack("{}4I".format(e), f.read(16)))
    dat1 = [struct.unpack("{}4f".format(e), f.read(16)) for _ in range(num_meshes)]
    mesh_blocks_info = []
    for i in range(num_meshes):
        flags, = struct.unpack("{}I".format(e), f.read(4))
        val1 = struct.unpack("{}4I".format(e), f.read(16)) # start of v1 - mesh, submesh, node, material
        uv_offset, = struct.unpack("{}I".format(e), f.read(4))
        idx_offset, = struct.unpack("{}I".format(e), f.read(4))
        vert_offset = read_offset(f)
        val2 = struct.unpack("{}5I".format(e), f.read(20)) # start of v2 - uv_stride, flags2, #verts, #idx, a zero
        name = read_string(f, read_offset(f))
        name_end_offset = read_offset(f)
        dat = {'flags': flags, 'name': name, 'mesh': val1[0], 'submesh': val1[1], 'node': val1[2],
            'material_id': val1[3], 'uv_offset': uv_offset + uv_start_offset, 'idx_offset': idx_offset + uv_start_offset,
            'vert_offset': vert_offset, 'num_verts': num_verts[i], 'uv_stride': val2[0], 'flags2': val2[1],
            'total_verts': val2[2], 'total_idx': val2[3], 'unk': val2[4]}
        mesh_blocks_info.append(dat)
    bone_palette_ids = struct.unpack("{}{}I".format(e, palette_count), f.read(4 * palette_count))
    meshes = []
    for i in range(num_meshes):
        meshes.append(read_mesh_wii(mesh_blocks_info[i], f))
    return(meshes, bone_palette_ids, mesh_blocks_info)

# Thank you to Dolphin emu team (https://github.com/dolphin-emu/dolphin)
# Although the implementation is different, this algorithm derived from Dolphin, licensed GPLv3
def decode_cmpr_block (f, image_array, img_start, pitch):
    def decode_565 (raw_color):
        r1, g1, b1 = (raw_color >> 11) & 0x1F, (raw_color >> 5) & 0x3F, raw_color & 0x1F
        r2, g2, b2 = r1 << 3 | r1 >> 2, g1 << 2 | g1 >> 4, b1 << 3 | b1 >> 2
        return([r2, g2, b2, 255])
    def DXTBlend (byte1, byte2):
        return ((byte1 * 3 + byte2 * 5) >> 3)
    colors = [[0,0,0,0]]*4
    c1, c2 = struct.unpack(">2H", f.read(4))
    colors[0], colors[3] = decode_565(c1), decode_565(c2)
    if c1 > c2:
        colors[1] = [DXTBlend(colors[3][0], colors[0][0]), DXTBlend(colors[3][1], colors[0][1]), DXTBlend(colors[3][2], colors[0][2]), 255]
        colors[2] = [DXTBlend(colors[0][0], colors[3][0]), DXTBlend(colors[0][1], colors[3][1]), DXTBlend(colors[0][2], colors[3][2]), 255]
    else:
        colors[1] = [(colors[0][0] + colors[3][0])//2, (colors[0][1] + colors[3][1])//2, (colors[0][2] + colors[3][2])//2, 255]
        colors[2] = [(colors[0][0] + colors[3][0])//2, (colors[0][1] + colors[3][1])//2, (colors[0][2] + colors[3][2])//2, 0]
    colors = [colors[0], colors[3], colors[1], colors[2]]
    lines = struct.unpack(">4B", f.read(4))
    pixel = img_start
    for y in range(4):
        val = lines[y]
        for x in range(4):
            image_array[pixel + x] = colors[(val >> 6) & 3]
            val <<= 2
        pixel += pitch
    return(image_array)

def decode_cmpr_mipmap (f, width, height):
    image_array = [[0,0,0,0]]*(width * height)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            image_array = decode_cmpr_block (f, image_array, ((y * width) + x), width)
            image_array = decode_cmpr_block (f, image_array, ((y * width) + x + 4), width)
            image_array = decode_cmpr_block (f, image_array, (((y + 4) * width) + x), width)
            image_array = decode_cmpr_block (f, image_array, (((y + 4) * width) + x + 4), width)
    return(image_array)

def read_texture_section_wii (f, start_offset, tex_data_offset, tex_data_block_size):
    known_tex_types = [0xE, 0x2000E, 0x4000E]
    f.seek(start_offset)
    header = struct.unpack("{}6I".format(e), f.read(24)) #unk0, size, unk1, num_tex, unk, unk
    tex_data = []
    for _ in range(header[3]):
        data = struct.unpack("{}4I".format(e), f.read(16))
        name = read_string(f, read_offset(f))
        tex_offset, = struct.unpack("{}I".format(e), f.read(4))
        unk, = struct.unpack("{}I".format(e), f.read(4))
        tex_data.append({'name': name, 'dwWidth': data[0], 'dwHeight': data[1], 'dwMipMapCount': 1,
            'type': 'BGRA32' if data[3] in known_tex_types else 'RAW', 'type_code': data[3],
            'offset': tex_offset + tex_data_offset, 'unk': unk})
    textures = []
    for i in range(len(tex_data)):
        if i < (len(tex_data) - 1):
            size = tex_data[i+1]['offset'] - tex_data[i]['offset']
        else:
            size = (tex_data_offset + tex_data_block_size) - tex_data[i]['offset']
        f.seek(tex_data[i]['offset'])
        if tex_data[i]['type_code'] & 0xF == 0xE:
            pixels = decode_cmpr_mipmap(f, tex_data[i]['dwWidth'], tex_data[i]['dwHeight']) # [(R,G,B,A), ...]
            raw_color_vals = [x[2] + (x[1] << 8) + (x[0] << 16) + (x[3] << 24) for x in pixels] # BGRA
            raw_data = struct.pack("<{}I".format(len(raw_color_vals)), *raw_color_vals)
        else:
            print("Missing format! {} is in format {}".format(tex_data[i]['name'], hex(tex_data[i]['type_code'])))
            raw_data = f.read(size)
        textures.append({'name': "{0}.{1}".format(tex_data[i]['name'], 'raw' if tex_data[i]['type'] == 'RAW' else 'dds'),
            'data': raw_data if tex_data[i]['type'] == 'RAW' else make_dds_header(tex_data[i]) + raw_data})
    return(textures)

def process_model_wii (f, base_name, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    global e
    print("Processing {}...".format(base_name))
    e = '>'
    set_endianness('>') # Figure out later how to determine this
    magic = f.read(4)
    if magic == b'FPS4':
        header = struct.unpack("{}6I".format(e), f.read(24))
        toc = []
        for i in range(header[0]):
            toc.append(struct.unpack("{}3I".format(e), f.read(12))) # offset, padded length, true length
        vgmaps = []
        magic2 = f.read(4)
        if magic2 == b'sing': # Single model mode
            skel_struct = read_skel_section (f, toc[0][0])
            meshes, bone_palette_ids, mesh_blocks_info = read_mesh_section_wii(f, toc[1][0], toc[2][0])
            material_struct = read_material_section (f, toc[5][0])
            mesh_blocks_info = material_id_to_index(mesh_blocks_info, material_struct, 0)
            vgmap = {'bone_{}'.format(bone_palette_ids[i]):i for i in range(len(bone_palette_ids))}
            if not all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                meshes, bone_palette_ids = repair_mesh_weights(meshes, bone_palette_ids, skel_struct)
            if all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                skel_index = {skel_struct[i]['id']:i for i in range(len(skel_struct))}
                vgmap = {skel_struct[skel_index[bone_palette_ids[i]]]['name']:i for i in range(len(bone_palette_ids))}
            for i in range(len(mesh_blocks_info)):
                mesh_blocks_info[i]['vgmap'] = len(vgmaps)
            vgmaps.append(vgmap)
            textures = read_texture_section_wii(f, toc[7][0], toc[8][0], toc[8][2])
        elif magic2 == b'mult': # Multi model mode
            meshes, mesh_blocks_info, material_struct, textures = [], [], [], []
            for i in range(len(toc)):
                f.seek(toc[i][0])
                dat = f.read(toc[i][2])
                with io.BytesIO(dat) as f2:
                    magic_i = f2.read(4)
                    if magic_i == b'FPS4':
                        header_i = struct.unpack("{}6I".format(e), f2.read(24))
                        toc_i = []
                        for i in range(header_i[0]):
                            toc_i.append(struct.unpack("{}3I".format(e), f2.read(12))) # offset, padded length, true length
                        magic2_i = f2.read(4)
                        if magic2_i == b'base': # Skeleton mode
                            skel_struct = read_skel_section (f2, toc_i[0][0])
                        elif magic2_i == b'part':
                            meshes_i, bone_palette_ids, mesh_blocks_info_i = read_mesh_section_wii(f2, toc_i[0][0], toc_i[1][0])
                            material_struct_i = read_material_section (f2, toc_i[4][0])
                            mesh_blocks_info_i = material_id_to_index(mesh_blocks_info_i, material_struct_i, len(material_struct))
                            meshes.extend(meshes_i)
                            material_struct.extend(material_struct_i)
                            vgmap = {'bone_{}'.format(bone_palette_ids[i]):i for i in range(len(bone_palette_ids))}
                            if not all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                                meshes, bone_palette_ids = repair_mesh_weights(meshes, bone_palette_ids, skel_struct)
                            if all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                                skel_index = {skel_struct[i]['id']:i for i in range(len(skel_struct))}
                                vgmap = {skel_struct[skel_index[bone_palette_ids[i]]]['name']:i for i in range(len(bone_palette_ids))}
                            for i in range(len(mesh_blocks_info_i)):
                                mesh_blocks_info_i[i]['vgmap'] = len(vgmaps)
                            vgmaps.append(vgmap)
                            mesh_blocks_info.extend(mesh_blocks_info_i)
                            textures.extend(read_texture_section_wii(f2, toc_i[5][0], toc_i[6][0], toc_i[6][2]))
        else:
            print("{} is not the expected model format, skipping!".format(base_name))
            return
        tex_overwrite = True # if overwrite == True else False
        if os.path.exists('textures') and (os.path.isdir('textures')) and (tex_overwrite == False):
            if str(input("'textures' folder exists! Overwrite? (y/N) ")).lower()[0:1] == 'y':
                tex_overwrite = True
        if (tex_overwrite == True) or not os.path.exists('textures'):
            if not os.path.exists('textures'):
                os.mkdir('textures')
            for i in range(len(textures)):
                open('textures/' + textures[i]['name'], 'wb').write(textures[i]['data'])
        write_gltf(base_name, skel_struct, vgmaps, mesh_blocks_info, meshes, material_struct,\
            overwrite = overwrite, write_binary_gltf = write_binary_gltf)
        if write_raw_buffers == True:
            if os.path.exists(base_name) and (os.path.isdir(base_name)) and (overwrite == False):
                if str(input(base_name + " folder exists! Overwrite? (y/N) ")).lower()[0:1] == 'y':
                    overwrite = True
            if (overwrite == True) or not os.path.exists(base_name):
                if not os.path.exists(base_name):
                    os.mkdir(base_name)
                for i in range(len(meshes)):
                    filename = '{0:02d}_{1}'.format(i, mesh_blocks_info[i]['name'])
                    write_fmt(meshes[i]['fmt'], '{0}/{1}.fmt'.format(base_name, filename))
                    write_ib(meshes[i]['ib'], '{0}/{1}.ib'.format(base_name, filename), meshes[i]['fmt'], '<')
                    write_vb(meshes[i]['vb'], '{0}/{1}.vb'.format(base_name, filename), meshes[i]['fmt'], '<')
                    open('{0}/{1}.vgmap'.format(base_name, filename), 'wb').write(
                        json.dumps(vgmaps[mesh_blocks_info[i]['vgmap']],indent=4).encode())
                mesh_struct = [{y:x[y] for y in x if not any(
                    ['offset' in y, 'num' in y])} for x in mesh_blocks_info]
                for i in range(len(mesh_struct)):
                    mesh_struct[i]['material'] = material_struct[mesh_struct[i]['material']]['name']
                mesh_struct = [{'id_referenceonly': i, **mesh_struct[i]} for i in range(len(mesh_struct))]
                write_struct_to_json(mesh_struct, base_name + '/mesh_info')
                write_struct_to_json(material_struct, base_name + '/material_info')
                #write_struct_to_json(skel_struct, base_name + '/skeleton_info')
    return True

def process_mdl_wii (mdl_file, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    base_name = mdl_file[:-4]
    with open(mdl_file, 'rb') as f:
        process_model_wii (f, base_name, overwrite = overwrite, write_raw_buffers = write_raw_buffers, write_binary_gltf = write_binary_gltf)

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
        parser.add_argument('mdl_file', help="Name of model file to process.")
        args = parser.parse_args()
        if os.path.exists(args.mdl_file) and args.mdl_file[-4:].upper() == '.MDL':
            process_mdl_wii(args.mdl_file, overwrite = args.overwrite, \
                write_raw_buffers = args.dumprawbuffers, write_binary_gltf = args.textformat)
    else:
        mdl_files = glob.glob('*.MDL')
        for mdl_file in mdl_files:
            process_mdl_wii(mdl_file)
