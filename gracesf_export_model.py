# Tool to export model data from the model format used by Tales of Graces f (PS3).
#
# Usage:  Run by itself without commandline arguments and it will search for model files
# and export a .gib file.
#
# For command line options, run:
# /path/to/python3 gracesf_export_model.py --help
#
# Requires lib_fmtibvb.py, put in the same directory
#
# GitHub eArmada8/gracesf_model_tool

try:
    import struct, json, numpy, copy, glob, os, sys
    from lib_fmtibvb import *
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

# Global variable, do not edit
e = '<'

def set_endianness (endianness):
    global e
    if endianness in ['<', '>']:
        e = endianness
    return

def read_offset (f):
    start_offset = f.tell()
    diff_offset, = struct.unpack("{}I".format(e), f.read(4))
    return(start_offset + diff_offset)

def read_string (f, start_offset):
    current_loc = f.tell()
    f.seek(start_offset)
    null_term_string = f.read(1)
    while null_term_string[-1] != 0:
        null_term_string += f.read(1)
    f.seek(current_loc)
    return(null_term_string[:-1].decode())

def trianglestrip_to_list(ib_list):
    triangles = []
    split_lists = [[]]
    # Split ib_list by primitive restart command, some models have this
    for i in range(len(ib_list)):
        if not ib_list[i] == -1:
            split_lists[-1].append(ib_list[i])
        else:
            split_lists.append([])
    for i in range(len(split_lists)):
        for j in range(len(split_lists[i])-2):
            if j % 2 == 0:
                triangles.append([split_lists[i][j], split_lists[i][j+1], split_lists[i][j+2]])
            else:
                triangles.append([split_lists[i][j], split_lists[i][j+2], split_lists[i][j+1]]) #DirectX implementation
                #triangles.append([split_lists[i][j+1], split_lists[i][j], split_lists[i][j+2]]) #OpenGL implementation
    # Remove degenerate triangles
    triangles = [x for x in triangles if len(set(x)) == 3]
    return(triangles)

def read_skel_section (f, start_offset):
    f.seek(start_offset)
    header = struct.unpack("{}6I".format(e), f.read(24)) #unk0, size, unk1, num_bones, unk2, unk3, unk4
    mtx_offset = read_offset(f)
    num_bones = header[3]
    skel_struct = []
    dat0 = list(struct.unpack("{}{}I".format(e, num_bones), f.read(num_bones * 4))) # list of bones
    bone_id_dict = {dat0[i]:i for i in range(len(dat0))}
    dat1 = []
    for i in range(num_bones):
        dat = {'id': dat0[i]}
        dat1a = struct.unpack("{}6i".format(e), f.read(24))
        dat['name'] = read_string (f, read_offset(f))
        offset2 = read_offset(f) # String end offset
        dat['parent'] = bone_id_dict[dat1a[1]] if dat1a[1] in bone_id_dict else dat1a[1] # Maybe should be -1 as default
        dat1.append(dat1a)
        skel_struct.append(dat)
    #Skip giant section of floats, then names of bones
    f.seek(mtx_offset)
    inv_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(num_bones)] # Stored correctly
    abs_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(num_bones)] # Stored transposed
    abs_mtx_flip = [numpy.array(abs_mtx[i]).reshape(4,4).flatten('F').tolist() for i in range(len(abs_mtx))] # Column major
    for i in range(num_bones):
        skel_struct[i]['abs_matrix'] = abs_mtx_flip[i]
        skel_struct[i]['inv_matrix'] = inv_mtx[i]
    for i in range(len(skel_struct)):
        if skel_struct[i]['parent'] in range(len(skel_struct)):
            abs_mtx = [skel_struct[i]['abs_matrix'][0:4], skel_struct[i]['abs_matrix'][4:8],\
                skel_struct[i]['abs_matrix'][8:12], skel_struct[i]['abs_matrix'][12:16]]
            parent_inv_mtx = [skel_struct[skel_struct[i]['parent']]['inv_matrix'][0:4],\
                skel_struct[skel_struct[i]['parent']]['inv_matrix'][4:8],\
                skel_struct[skel_struct[i]['parent']]['inv_matrix'][8:12],\
                skel_struct[skel_struct[i]['parent']]['inv_matrix'][12:16]]
            skel_struct[i]['matrix'] = numpy.dot(abs_mtx, parent_inv_mtx).flatten('C').tolist()
        else:
            skel_struct[i]['matrix'] = skel_struct[i]['abs_matrix']
        skel_struct[i]['children'] = [j for j in range(len(skel_struct)) if skel_struct[j]['parent'] == i]
    return(skel_struct)

def make_fmt(num_uvs, has_weights = True):
    fmt = {'stride': '0', 'topology': 'trianglelist', 'format':\
        "DXGI_FORMAT_R16_UINT", 'elements': []}
    element_id, stride = 0, 0
    semantic_index = {'TEXCOORD': 0} # Counters for multiple indicies
    elements = []
    for i in range(2 + num_uvs + (2 if has_weights else 0)):
            # I think order matters in this dict, so we will define the entire structure with default values
            element = {'id': '{0}'.format(element_id), 'SemanticName': '', 'SemanticIndex': '0',\
                'Format': '', 'InputSlot': '0', 'AlignedByteOffset': str(stride),\
                'InputSlotClass': 'per-vertex', 'InstanceDataStepRate': '0'}
            if i == 0:
                element['SemanticName'] = 'POSITION'
                element['Format'] = 'R32G32B32_FLOAT'
                stride += 12
            elif i == 1:
                element['SemanticName'] = 'NORMAL'
                element['Format'] = 'R32G32B32_FLOAT'
                stride += 12
            elif i == 4+num_uvs-2:
                element['SemanticName'] = 'BLENDWEIGHTS'
                element['Format'] = 'R32G32B32A32_FLOAT'
                stride += 16
            elif i == 4+num_uvs-1:
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

def read_mesh (mesh_info, f):
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
    f.seek(mesh_info['idx_offset'])
    count, = struct.unpack("{}I".format(e), f.read(4))
    # sub_counts is (number of vertices, number of indices) per count
    sub_counts = [struct.unpack("{}2H".format(e), f.read(4)) for _ in range(count)]
    # Indices
    idx_buffer = []
    for i in range(count):
        idx_subbuffer = list(struct.unpack("{0}{1}h".format(e, sub_counts[i][1]), f.read(sub_counts[i][1] * 2)))
        if i > 0:
            idx_subbuffer = [x+sum([x[0] for x in sub_counts][0:i]) if not x == -1 else x for x in idx_subbuffer]
        idx_buffer.extend(idx_subbuffer)
        if i < (count - 1):
            idx_buffer.append(-1)
    # Vertices
    f.seek(mesh_info['vert_offset'])
    uv_stride = mesh_info['uv_stride']
    num_uv_maps = mesh_info['flags2'] & 0xF
    verts = []
    norms = []
    if mesh_info['flags'] & 0xF00 == 0x100:
        blend_idx = []
        weights = []
        num_verts = mesh_info['num_verts']
        total_verts = sum(num_verts)
        for i in range(count): # should always be 1 here I think
            for j in range(len(num_verts)):
                vert_offset = f.tell()
                norm_offset = vert_offset + 12
                blend_idx_offset = norm_offset + 12
                weights_offset = blend_idx_offset + 4
                stride = 28 + (j * 4)
                end_offset = f.tell() + (num_verts[j] * stride)
                f.seek(vert_offset)
                verts.extend(read_interleaved_floats(f, 3, stride, num_verts[j]))
                f.seek(norm_offset)
                norms.extend(read_interleaved_floats(f, 3, stride, num_verts[j]))
                f.seek(blend_idx_offset)
                blend_idx.extend(read_interleaved_bytes(f, 4, stride, num_verts[j]))
                if j > 0:
                    f.seek(weights_offset)
                    weights.extend(read_interleaved_floats(f, j, stride, num_verts[j]))
                else:
                    weights.extend([[1.0] for _ in range(num_verts[j])])
                f.seek(end_offset)
            weights = fix_weights(weights)
            if i < (count - 1):
                num_verts = struct.unpack("{}4I".format(e), f.read(16))
                total_verts += sum(num_verts)
    elif mesh_info['flags'] & 0xF00 == 0x400:
        total_verts = mesh_info['total_verts']
        vert_offset = mesh_info['uv_offset']
        norm_offset = vert_offset + 12
        stride = uv_stride
        f.seek(vert_offset)
        verts.extend(read_interleaved_floats(f, 3, stride, total_verts))
        f.seek(norm_offset)
        norms.extend(read_interleaved_floats(f, 3, stride, total_verts))
    elif mesh_info['flags'] & 0xF00 == 0x700:
        # No weights, so only mesh_info['num_verts'][0] is non-zero
        total_verts = sum(mesh_info['num_verts'])
        vert_offset = f.tell()
        verts.extend([read_floats(f, 3) for _ in range(total_verts)])
        norms.extend([read_floats(f, 3) for _ in range(total_verts)])
    uv_maps = []
    if mesh_info['flags'] & 0xF00 in [0x100, 0x700]:
        for i in range(num_uv_maps):
            f.seek(mesh_info['uv_offset'] + 4 + (i * 8))
            uv_maps.append(read_interleaved_floats (f, 2, uv_stride, total_verts))
    elif mesh_info['flags'] & 0xF00 == 0x400:
        for i in range(num_uv_maps):
            f.seek(mesh_info['uv_offset'] + 28 + (i * 8))
            uv_maps.append(read_interleaved_floats (f, 2, uv_stride, total_verts))
    fmt = make_fmt(len(uv_maps), True)
    vb = [{'Buffer': verts}, {'Buffer': norms}]
    for uv_map in uv_maps:
        vb.append({'Buffer': uv_map})
    if mesh_info['flags'] & 0xF00 == 0x100:
        vb.append({'Buffer': weights})
        vb.append({'Buffer': blend_idx})
    elif mesh_info['flags'] & 0xF00 in [0x400, 0x700]:
        vb.append({'Buffer': [[1.0, 0.0, 0.0, 0.0] for _ in range(len(verts))]})
        vb.append({'Buffer': [[0, 0, 0, 0] for _ in range(len(verts))]})
    return({'fmt': fmt, 'vb': vb, 'ib': trianglestrip_to_list(idx_buffer)})

def read_mesh_section (f, start_offset, uv_start_offset):
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
        idx_offset = read_offset(f)
        vert_offset = read_offset(f)
        val2 = struct.unpack("{}5I".format(e), f.read(20)) # start of v2 - uv_stride, flags2, #verts, #idx, a zero
        name = read_string(f, read_offset(f))
        name_end_offset = read_offset(f)
        dat = {'flags': flags, 'name': name, 'mesh': val1[0], 'submesh': val1[1], 'node': val1[2],
            'material_id': val1[3], 'uv_offset': uv_offset + uv_start_offset, 'idx_offset': idx_offset,
            'vert_offset': vert_offset, 'num_verts': num_verts[i], 'uv_stride': val2[0], 'flags2': val2[1],
            'total_verts': val2[2], 'total_idx': val2[3], 'unk': val2[4]}
        mesh_blocks_info.append(dat)
    bone_palette_ids = struct.unpack("{}{}I".format(e, palette_count), f.read(4 * palette_count))
    meshes = []
    for i in range(num_meshes):
        meshes.append(read_mesh(mesh_blocks_info[i], f))
    return(meshes, bone_palette_ids, mesh_blocks_info)

def repair_mesh_weights (meshes, bone_palette_ids, skel_struct):
    used_groups_buffer = []
    for i in range(len(meshes)):
        for j in range(len(meshes[i]['vb'][0]['Buffer'])):
            total, k = 0.0, 0
            while total < 1.0 and k < 4:
                total += meshes[i]['vb'][-2]['Buffer'][j][k]
                k += 1
            used_groups_buffer.extend(meshes[i]['vb'][-1]['Buffer'][j][:k])
    used_groups = sorted(list(set(used_groups_buffer)))
    new_palette_ids = [bone_palette_ids[i] for i in range(len(bone_palette_ids)) if i in used_groups]
    if all([y in [x['id'] for x in skel_struct] for y in new_palette_ids]):
        old_to_new = {used_groups[i]:i for i in range(len(used_groups))}
        for i in range(len(meshes)):
            new_wt_buffer = []
            for j in range(len(meshes[i]['vb'][0]['Buffer'])):
                new_wt = []
                total, k = 0.0, 0
                while total < 1.0 and k < 4:
                    total += meshes[i]['vb'][-2]['Buffer'][j][k]
                    new_wt.append(old_to_new[meshes[i]['vb'][-1]['Buffer'][j][k]])
                    k += 1
                while k < 4:
                    new_wt.append(0)
                    k += 1
                new_wt_buffer.append(new_wt)
            meshes[i]['vb'][-1]['Buffer'] = new_wt_buffer
        return(meshes, new_palette_ids)
    else:
        return(meshes, bone_palette_ids) # Could not remap, return original values

def read_material_section (f, start_offset):
    f.seek(start_offset)
    header = struct.unpack("{}5I".format(e), f.read(20)) #unk0, size, unk1, num_mats, maybe num_tex?
    num_materials = header[3]
    set_0 = []
    for _ in range(num_materials):
        num_tex, internal_id = struct.unpack("{}2i".format(e), f.read(8))
        data0 = list(struct.unpack("{}if4i".format(e), f.read(24)))
        if num_tex > 0:
            more_ints = (num_tex) * 2
            data0.extend(struct.unpack("{}{}i".format(e, more_ints), f.read(more_ints * 4)))
        set_0.append({'num_tex': num_tex, 'internal_id': internal_id, 'data0': data0})
    unk, = struct.unpack("{}I".format(e), f.read(4))
    set_1 = []
    for i in range(num_materials):
        name = read_string(f, read_offset(f))
        end_offset = read_offset(f)
        unk1, = struct.unpack("{}I".format(e), f.read(4))
        tex_names = []
        for _ in range(set_0[i]['num_tex']):
            tex_name = read_string(f, read_offset(f))
            tex_val, = struct.unpack("{}I".format(e), f.read(4))
            tex_names.append([tex_name, tex_val])
        unk2 = struct.unpack("{}I".format(e), f.read(4))
        set_1.append({'name': name, 'tex_names': tex_names, 'unk': [unk1, unk2]})
    material_struct = []
    if len(set_0) == len(set_1):
        for i in range(len(set_0)):
            material = {'name': set_1[i]['name']}
            material['textures'] = [x[0] for x in set_1[i]['tex_names']]
            material['alpha'] = 1 #need to work this out
            material['internal_id'] = set_0[i]['internal_id']
            material['unk_parameters'] = {'set_0': set_0[i]['data0'], 'set_1': set_1[i]['unk']}
            material_struct.append(material)
    return (material_struct)

def material_id_to_index (mesh_blocks_info, material_struct, offset):
    material_dict = {material_struct[i]['internal_id']: i + offset for i in range(len(material_struct))}
    for i in range(len(mesh_blocks_info)):
        mesh_blocks_info[i]['material'] = material_dict[mesh_blocks_info[i]['material_id']]
    return(mesh_blocks_info)

def make_dds_header (tex_info):
    if tex_info['type'] == 'DXT1':
        dwPitchOrLinearSize = (tex_info['dwHeight'] // 4) * (tex_info['dwWidth'] // 4) * 8
        pixel_format = {'dwSize': 32, 'dwFlags': 0x4, 'dwFourCC': 'DXT1', 'dwRGBBitCount': 0,
            'dwRBitMask': 0, 'dwGBitMask': 0, 'dwBBitMask': 0, 'dwABitMask': 0}
    elif tex_info['type'] == 'DXT5':
        dwPitchOrLinearSize = (tex_info['dwHeight'] // 4) * (tex_info['dwWidth'] // 4) * 16
        pixel_format = {'dwSize': 32, 'dwFlags': 0x4, 'dwFourCC': 'DXT5', 'dwRGBBitCount': 0,
            'dwRBitMask': 0, 'dwGBitMask': 0, 'dwBBitMask': 0, 'dwABitMask': 0}
    elif tex_info['type'] == 'BGRA32':
        dwPitchOrLinearSize = (tex_info['dwWidth'] * 32 + 7) // 8
        pixel_format = {'dwSize': 32, 'dwFlags': 0x41, 'dwFourCC': '\u0000\u0000\u0000\u0000', 'dwRGBBitCount': 32,
            'dwRBitMask': 0xFF0000, 'dwGBitMask': 0xFF00, 'dwBBitMask': 0xFF, 'dwABitMask': 0xFF000000}
    else:
        return(b'')
    header_info = {'dwSize': 124, 'dwFlags': 0xA1007, 'dwHeight': tex_info['dwHeight'],\
        'dwWidth': tex_info['dwWidth'], 'dwPitchOrLinearSize': dwPitchOrLinearSize,\
        'dwDepth': 1, 'dwMipMapCount': tex_info['dwMipMapCount'], 'pixel_format': pixel_format, 'dwCaps': 0x401008,\
        'dwCaps2': 0, 'dwCaps3': 0, 'dwCaps4': 0}
    header = b'DDS ' + struct.pack("<18I", *([header_info['dwSize'], header_info['dwFlags'],\
        header_info['dwHeight'], header_info['dwWidth'], header_info['dwPitchOrLinearSize'],\
        header_info['dwDepth'], header_info['dwMipMapCount']] + [0 for x in range(11)]))
    header += struct.pack("<2I", header_info["pixel_format"]['dwSize'], header_info["pixel_format"]['dwFlags'])
    header += header_info["pixel_format"]['dwFourCC'].encode()
    header += struct.pack("<5I", header_info["pixel_format"]['dwRGBBitCount'], header_info["pixel_format"]['dwRBitMask'],\
        header_info["pixel_format"]['dwGBitMask'], header_info["pixel_format"]['dwBBitMask'],\
        header_info["pixel_format"]['dwABitMask'])
    header += struct.pack("<5I", header_info['dwCaps'], header_info['dwCaps2'], header_info['dwCaps3'],\
        header_info['dwCaps4'], 0)
    return(header)

def endian_swap_argb (raw_tex_data):
    len_ = len(raw_tex_data) // 4
    vals = struct.unpack(">{}I".format(len_), raw_tex_data)
    return(struct.pack("<{}I".format(len_), *vals))

def read_texture_section (f, start_offset, tex_data_offset, tex_data_block_size):
    known_tex_types = {0x85: 'BGRA32', 0x86: 'DXT1', 0x88: 'DXT5',
        0xa5: 'BGRA32', 0xa6: 'DXT1', 0xa8: 'DXT5'}
    f.seek(start_offset)
    header = struct.unpack("{}6I".format(e), f.read(24)) #unk0, size, unk1, num_tex, unk, unk
    tex_data = []
    for _ in range(header[3]):
        data = struct.unpack("{}4I".format(e), f.read(16))
        name = read_string(f, read_offset(f))
        tex_offset, = struct.unpack("{}I".format(e), f.read(4))
        unk, = struct.unpack("{}I".format(e), f.read(4))
        type_ = data[3] >> 24
        tex_data.append({'name': name, 'dwWidth': data[0], 'dwHeight': data[1], 'dwMipMapCount': data[2],
            'type': known_tex_types[type_] if type_ in known_tex_types else 'RAW',
            'offset': tex_offset + tex_data_offset, 'unk': unk})
    textures = []
    for i in range(len(tex_data)):
        if i < (len(tex_data) - 1):
            size = tex_data[i+1]['offset'] - tex_data[i]['offset']
        else:
            size = (tex_data_offset + tex_data_block_size) - tex_data[i]['offset']
        f.seek(tex_data[i]['offset'])
        raw_data = f.read(size)
        if e == '>' and tex_data[i]['type'] == 'BGRA32':
            raw_data = endian_swap_argb (raw_data)
        textures.append({'name': "{0}.{1}".format(tex_data[i]['name'], 'raw' if tex_data[i]['type'] == 'RAW' else 'dds'),
            'data': raw_data if tex_data[i]['type'] == 'RAW' else make_dds_header(tex_data[i]) + raw_data})
    return(textures)

def convert_format_for_gltf(dxgi_format):
    dxgi_format = dxgi_format.split('DXGI_FORMAT_')[-1]
    dxgi_format_split = dxgi_format.split('_')
    if len(dxgi_format_split) == 2:
        numtype = dxgi_format_split[1]
        vec_format = re.findall("[0-9]+",dxgi_format_split[0])
        vec_bits = int(vec_format[0])
        vec_elements = len(vec_format)
        if numtype in ['FLOAT', 'UNORM', 'SNORM']:
            componentType = 5126
            componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
            dxgi_format = "".join(['R32','G32','B32','A32','D32'][0:componentStride//4]) + "_FLOAT"
        elif numtype == 'UINT':
            if vec_bits == 32:
                componentType = 5125
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
            elif vec_bits == 16:
                componentType = 5123
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 2
            elif vec_bits == 8:
                componentType = 5121
                componentStride = len(re.findall('[0-9]+', dxgi_format))
        accessor_types = ["SCALAR", "VEC2", "VEC3", "VEC4"]
        accessor_type = accessor_types[len(re.findall('[0-9]+', dxgi_format))-1]
        return({'format': dxgi_format, 'componentType': componentType,\
            'componentStride': componentStride, 'accessor_type': accessor_type})
    else:
        return(False)

def convert_fmt_for_gltf(fmt):
    new_fmt = copy.deepcopy(fmt)
    stride = 0
    new_semantics = {'BLENDWEIGHTS': 'WEIGHTS', 'BLENDINDICES': 'JOINTS'}
    need_index = ['WEIGHTS', 'JOINTS', 'COLOR', 'TEXCOORD']
    for i in range(len(fmt['elements'])):
        if new_fmt['elements'][i]['SemanticName'] in new_semantics.keys():
            new_fmt['elements'][i]['SemanticName'] = new_semantics[new_fmt['elements'][i]['SemanticName']]
        new_info = convert_format_for_gltf(fmt['elements'][i]['Format'])
        new_fmt['elements'][i]['Format'] = new_info['format']
        if new_fmt['elements'][i]['SemanticName'] in need_index:
            new_fmt['elements'][i]['SemanticName'] = new_fmt['elements'][i]['SemanticName'] + '_' +\
                new_fmt['elements'][i]['SemanticIndex']
        new_fmt['elements'][i]['AlignedByteOffset'] = stride
        new_fmt['elements'][i]['componentType'] = new_info['componentType']
        new_fmt['elements'][i]['componentStride'] = new_info['componentStride']
        new_fmt['elements'][i]['accessor_type'] = new_info['accessor_type']
        stride += new_info['componentStride']
    index_fmt = convert_format_for_gltf(fmt['format'])
    new_fmt['format'] = index_fmt['format']
    new_fmt['componentType'] = index_fmt['componentType']
    new_fmt['componentStride'] = index_fmt['componentStride']
    new_fmt['accessor_type'] = index_fmt['accessor_type']
    new_fmt['stride'] = stride
    return(new_fmt)

def fix_strides(submesh):
    offset = 0
    for i in range(len(submesh['vb'])):
        submesh['vb'][i]['fmt']['AlignedByteOffset'] = str(offset)
        submesh['vb'][i]['stride'] = get_stride_from_dxgi_format(submesh['vb'][i]['fmt']['Format'])
        offset += submesh['vb'][i]['stride']
    return(submesh)

def write_gltf(base_name, skel_struct, vgmaps, mesh_blocks_info, meshes, material_struct,\
        overwrite = False, write_binary_gltf = True):
    gltf_data = {}
    gltf_data['asset'] = { 'version': '2.0' }
    gltf_data['accessors'] = []
    gltf_data['bufferViews'] = []
    gltf_data['buffers'] = []
    gltf_data['meshes'] = []
    gltf_data['materials'] = []
    gltf_data['nodes'] = []
    gltf_data['samplers'] = []
    gltf_data['scenes'] = [{}]
    gltf_data['scenes'][0]['nodes'] = [0]
    gltf_data['scene'] = 0
    gltf_data['skins'] = []
    gltf_data['textures'] = []
    giant_buffer = bytes()
    buffer_view = 0
    # Materials
    material_dict = [{'name': material_struct[i]['name'],
        'texture': material_struct[i]['textures'][0] if len(material_struct[i]['textures']) > 0 else '',
        'alpha': material_struct[i]['alpha'] if 'alpha' in material_struct[i] else 0}
        for i in range(len(material_struct))]
    texture_list = sorted(list(set([x['texture'] for x in material_dict if not x['texture'] == ''])))
    gltf_data['images'] = [{'uri':'textures/{}.dds'.format(x)} for x in texture_list]
    for mat in material_dict:
        material = { 'name': mat['name'] }
        material['pbrMetallicRoughness'] = { 'metallicFactor' : 0.0, 'roughnessFactor' : 1.0 }
        if not mat['texture'] == '':
            sampler = { 'wrapS': 10497, 'wrapT': 10497 } # I have no idea if this setting exists
            texture = { 'source': texture_list.index(mat['texture']), 'sampler': len(gltf_data['samplers']) }
            material['pbrMetallicRoughness']['baseColorTexture'] = { 'index' : len(gltf_data['textures']), }
            gltf_data['samplers'].append(sampler)
            gltf_data['textures'].append(texture)
        if mat['alpha'] & 1:
            material['alphaMode'] = 'MASK'
        gltf_data['materials'].append(material)
    material_list = [x['name'] for x in gltf_data['materials']]
    missing_textures = [x['uri'] for x in gltf_data['images'] if not os.path.exists(x['uri'])]
    if len(missing_textures) > 0:
        print("Warning:  The following textures were not found:")
        for texture in missing_textures:
            print("{}".format(texture))
    # Nodes
    for i in range(len(skel_struct)):
        g_node = {'children': skel_struct[i]['children'], 'name': skel_struct[i]['name'], 'matrix': skel_struct[i]['matrix']}
        gltf_data['nodes'].append(g_node)
    for i in range(len(gltf_data['nodes'])):
        if len(gltf_data['nodes'][i]['children']) == 0 and i > 0:
            del(gltf_data['nodes'][i]['children'])
    if len(gltf_data['nodes']) == 0:
        gltf_data['nodes'].append({'children': [], 'name': 'root'})
    # Mesh nodes will be attached to the first node since in the original model, they don't really have a home
    node_id_list = [x['id'] for x in skel_struct]
    for i in range(len(mesh_blocks_info)):
        mesh_blocks_info[i]['mesh_v'] = '{0}_{1:02d}'.format(mesh_blocks_info[i]['mesh'], mesh_blocks_info[i]['vgmap'])
    mesh_node_ids = {x['mesh_v']:x['name'] for x in mesh_blocks_info}
    if not 'children' in gltf_data['nodes'][0]:
        gltf_data['nodes'][0]['children'] = []
    for mesh_node_id in mesh_node_ids:
        if not mesh_node_id in node_id_list:
            g_node = {'name': mesh_node_ids[mesh_node_id]}
            gltf_data['nodes'][0]['children'].append(len(gltf_data['nodes']))
            gltf_data['nodes'].append(g_node)
    mesh_block_tree = {x:[i for i in range(len(mesh_blocks_info)) if mesh_blocks_info[i]['mesh_v'] == x] for x in mesh_node_ids}
    node_list = [x['name'] for x in gltf_data['nodes']]
    # Skin matrices
    skinning_possible = True
    try:
        ibms_struct = []
        inv_mtx_buffers = []
        for vgmap in vgmaps:
            vgmap_nodes = [node_list.index(x) for x in list(vgmap.keys())]
            ibms = [skel_struct[j]['inv_matrix'] for j in vgmap_nodes]
            inv_mtx_buffer = b''.join([struct.pack("<16f", *x) for x in ibms])
            ibms_struct.append(ibms)
            inv_mtx_buffers.append(inv_mtx_buffer)
    except ValueError:
        skinning_possible = False
    # Meshes
    mesh_names = [] # Xillia doesn't have a 
    for mesh in mesh_block_tree: #Mesh
        primitives = []
        for j in range(len(mesh_block_tree[mesh])): #Submesh
            i = mesh_block_tree[mesh][j]
            # Vertex Buffer
            gltf_fmt = convert_fmt_for_gltf(meshes[i]['fmt'])
            vb_stream = io.BytesIO()
            write_vb_stream(meshes[i]['vb'], vb_stream, gltf_fmt, e='<', interleave = False)
            block_offset = len(giant_buffer)
            primitive = {"attributes":{}}
            for element in range(len(gltf_fmt['elements'])):
                primitive["attributes"][gltf_fmt['elements'][element]['SemanticName']]\
                    = len(gltf_data['accessors'])
                gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                    "componentType": gltf_fmt['elements'][element]['componentType'],\
                    "count": len(meshes[i]['vb'][element]['Buffer']),\
                    "type": gltf_fmt['elements'][element]['accessor_type']})
                if gltf_fmt['elements'][element]['SemanticName'] == 'POSITION':
                    gltf_data['accessors'][-1]['max'] =\
                        [max([x[0] for x in meshes[i]['vb'][element]['Buffer']]),\
                         max([x[1] for x in meshes[i]['vb'][element]['Buffer']]),\
                         max([x[2] for x in meshes[i]['vb'][element]['Buffer']])]
                    gltf_data['accessors'][-1]['min'] =\
                        [min([x[0] for x in meshes[i]['vb'][element]['Buffer']]),\
                         min([x[1] for x in meshes[i]['vb'][element]['Buffer']]),\
                         min([x[2] for x in meshes[i]['vb'][element]['Buffer']])]
                gltf_data['bufferViews'].append({"buffer": 0,\
                    "byteOffset": block_offset,\
                    "byteLength": len(meshes[i]['vb'][element]['Buffer']) *\
                    gltf_fmt['elements'][element]['componentStride'],\
                    "target" : 34962})
                block_offset += len(meshes[i]['vb'][element]['Buffer']) *\
                    gltf_fmt['elements'][element]['componentStride']
            vb_stream.seek(0)
            giant_buffer += vb_stream.read()
            vb_stream.close()
            del(vb_stream)
            # Index Buffers
            ib_stream = io.BytesIO()
            write_ib_stream(meshes[i]['ib'], ib_stream, gltf_fmt, e='<')
            # IB is 16-bit so can be misaligned, unlike VB
            while (ib_stream.tell() % 4) > 0:
                ib_stream.write(b'\x00')
            primitive["indices"] = len(gltf_data['accessors'])
            gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                "componentType": gltf_fmt['componentType'],\
                "count": len([index for triangle in meshes[i]['ib'] for index in triangle]),\
                "type": gltf_fmt['accessor_type']})
            gltf_data['bufferViews'].append({"buffer": 0,\
                "byteOffset": len(giant_buffer),\
                "byteLength": ib_stream.tell(),\
                "target" : 34963})
            ib_stream.seek(0)
            giant_buffer += ib_stream.read()
            ib_stream.close()
            del(ib_stream)
            primitive["mode"] = 4 #TRIANGLES
            primitive["material"] = mesh_blocks_info[i]['material']
            primitives.append(primitive)
        if len(primitives) > 0:
            if mesh_node_ids[mesh] in node_list: # One of the new nodes
                node_id = node_list.index(mesh_node_ids[mesh])
            else: # One of the pre-assigned nodes
                node_id = node_id_list.index(mesh_blocks_info[i]["mesh_v"])
            gltf_data['nodes'][node_id]['mesh'] = len(gltf_data['meshes'])
            gltf_data['meshes'].append({"primitives": primitives, "name": mesh_node_ids[mesh]})
            # Skinning
            if len(vgmaps[mesh_blocks_info[i]["vgmap"]]) > 0 and skinning_possible == True:
                gltf_data['nodes'][node_id]['skin'] = len(gltf_data['skins'])
                gltf_data['skins'].append({"inverseBindMatrices": len(gltf_data['accessors']),\
                    "joints": [node_list.index(x) for x in vgmaps[mesh_blocks_info[i]["vgmap"]]]})
                gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                    "componentType": 5126,\
                    "count": len(ibms_struct[mesh_blocks_info[i]["vgmap"]]),\
                    "type": "MAT4"})
                gltf_data['bufferViews'].append({"buffer": 0,\
                    "byteOffset": len(giant_buffer),\
                    "byteLength": len(inv_mtx_buffers[mesh_blocks_info[i]["vgmap"]])})
                giant_buffer += inv_mtx_buffers[mesh_blocks_info[i]["vgmap"]]
    # Write GLB
    gltf_data['buffers'].append({"byteLength": len(giant_buffer)})
    if (os.path.exists(base_name + '.gltf') or os.path.exists(base_name + '.glb')) and (overwrite == False):
        if str(input(base_name + ".glb/.gltf exists! Overwrite? (y/N) ")).lower()[0:1] == 'y':
            overwrite = True
    if (overwrite == True) or not (os.path.exists(base_name + '.gltf') or os.path.exists(base_name + '.glb')):
        if write_binary_gltf == True:
            with open(base_name+'.glb', 'wb') as f:
                jsondata = json.dumps(gltf_data).encode('utf-8')
                jsondata += b' ' * (4 - len(jsondata) % 4)
                f.write(struct.pack('<III', 1179937895, 2, 12 + 8 + len(jsondata) + 8 + len(giant_buffer)))
                f.write(struct.pack('<II', len(jsondata), 1313821514))
                f.write(jsondata)
                f.write(struct.pack('<II', len(giant_buffer), 5130562))
                f.write(giant_buffer)
        else:
            gltf_data['buffers'][0]["uri"] = base_name+'.bin'
            with open(base_name+'.bin', 'wb') as f:
                f.write(giant_buffer)
            with open(base_name+'.gltf', 'wb') as f:
                f.write(json.dumps(gltf_data, indent=4).encode("utf-8"))

def process_model (f, base_name, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    print("Processing {}...".format(base_name))
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
            meshes, bone_palette_ids, mesh_blocks_info = read_mesh_section (f, toc[1][0], toc[2][0])
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
            textures = read_texture_section (f, toc[7][0], toc[8][0], toc[8][2])
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
                            meshes_i, bone_palette_ids, mesh_blocks_info_i = read_mesh_section (f2, toc_i[0][0], toc_i[1][0])
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
                            textures.extend(read_texture_section(f2, toc_i[5][0], toc_i[6][0], toc_i[6][2]))
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

def process_mdl (mdl_file, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    base_name = mdl_file[:-4]
    with open(mdl_file, 'rb') as f:
        process_model (f, base_name, overwrite = overwrite, write_raw_buffers = write_raw_buffers, write_binary_gltf = write_binary_gltf)

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
            process_mdl(args.mdl_file, overwrite = args.overwrite, \
                write_raw_buffers = args.dumprawbuffers, write_binary_gltf = args.textformat)
    else:
        mdl_files = glob.glob('*.MDL')
        for mdl_file in mdl_files:
            process_mdl(mdl_file)
