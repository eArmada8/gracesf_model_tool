# Tales of Graces f (PS3) mesh export
A script to get the mesh data out of the files from Tales of Graces f (PS3).  The output is in .glb files, although there is an option for .fmt/.ib/.vb/.vgmap that are compatible with DarkStarSword Blender import plugin for 3DMigoto.

## Credits:
I am as always very thankful for the dedicated reverse engineers at the Tales of ABCDE discord and the Kiseki modding discord, for their brilliant work, and for sharing that work so freely.  Thank you to NeXoGone and the original author of the noesis scripts for this game for structural information as well!  Thank you to Mc-muffin (github.com/Mc-muffin) for the python CompToE decompression implementation used unmodified in this tool, and thank you to Life Bottle (github.com/lifebottle) for writing the original CompToE implementation.

## Requirements:
1. Python 3.10 and newer is required for use of these scripts.  It is free from the Microsoft Store, for Windows users.  For Linux users, please consult your distro.
2. The numpy module for python is needed.  Install by typing "python3 -m pip install numpy" in the command line / shell.  (The struct, json, io, glob, copy, subprocess, os, sys, and argparse modules are also required, but these are all already included in most basic python installations.)
3. The output can be imported into Blender as .glb, or as raw buffers using DarkStarSword's amazing plugin: https://github.com/DarkStarSword/3d-fixes/blob/master/blender_3dmigoto.py (tested on commit [5fd206c](https://raw.githubusercontent.com/DarkStarSword/3d-fixes/5fd206c52fb8c510727d1d3e4caeb95dac807fb2/blender_3dmigoto.py))
4. `gracesf_export_model.py` is dependent on `lib_fmtibvb.py`, which must be in the same folder.  `gracesf_export_texfiles.py` and `gracesf_export_model_acf.py` are each dependent on both `gracesf_export_model.py` and `lib_fmtibvb.py`, which must be in the same folder.
5. [HyoutaTools](https://github.com/AdmiralCurtiss/HyoutaTools) can be used to unpack the .cpk archives (and subsequent .acf archives) that come with the game.  Unpacking .acf files with HyoutaTools is *optional* as gracesf_export_model_acf.py will process them directly.
6. OPTIONAL: [COMPTOE](https://github.com/lifebottle/comptoe) can be used to decompress the .MDL files (unpacked from .acf files from .cpk archives) prior to use with this script.  This is not needed if processing .acf files directly.

## Usage:
### gracesf_export_model.py
Double click the python script and it will search for all model files (decompressed .MDL files).  Textures will be placed in a `textures` folder.  *NOTE: `gracesf_export_model_acf.py` can be used instead to process .acf files directly, see below.*

**Command line arguments:**
`gracesf_export_model.py [-h] [-t] [-d] [-o] mdl_file`

`-t, --textformat`
Output .gltf/.bin format instead of .glb format.

`-d, --dumprawbuffers`
Dump .fmt/.ib/.vb/.vgmap files in a folder with the same name as the .mdl file.  Use DarkStarSword's plugin to view.

`-h, --help`
Shows help message.

`-o, --overwrite`
Overwrite existing files without prompting.

### gracesf_export_model_acf.py
Double click the python script and it will search for all model and texture .acf/.dat files and process them directly; any .MDL files and .TEX files within will be processed.  (Relevant files in the .cpk archives will have an .acf extension, but DLC acf files come with a .dat extension.)  Textures will be placed in a `textures` folder.  This script requires both gracesf_export_model.py and lib_fmtibvb.py to function.

**Command line arguments:**
`gracesf_export_model_acf.py [-h] [-t] [-d] [-o] acf_file`

`-t, --textformat`
Output .gltf/.bin format instead of .glb format.

`-d, --dumprawbuffers`
Dump .fmt/.ib/.vb/.vgmap files in a folder with the same name as the name of the internal .mdl file.  Use DarkStarSword's plugin to view.

`-h, --help`
Shows help message.

`-o, --overwrite`
Overwrite existing files without prompting.

### gracesf_export_texfiles.py
Double click the python script and it will search for all texture files (decompressed .TEX files).  Textures will be placed in a `textures` folder.  *Requires `gracesf_export_model.py` and `lib_fmtibvb.py` to be in the same folder due to shared decoding functions.*

**Command line arguments:**
`gracesf_export_texfiles.py [-h] tex_file`

`-h, --help`
Shows help message.