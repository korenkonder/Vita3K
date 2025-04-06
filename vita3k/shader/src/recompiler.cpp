// Vita3K emulator project
// Copyright (C) 2025 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include <shader/glsl/recompiler.h>
#include <shader/spirv/recompiler.h>
#include <shader/recompiler.h>
#include <util/fs.h>

namespace shader {

// ***************************
// * Functions (exposed API) *
// ***************************

GeneratedShader convert_gxp(const SceGxmProgram &program, const std::string &shader_name, const FeatureState &features, const Target target, const Hints &hints, bool maskupdate,
    bool force_shader_debug, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper) {
    if (target == Target::SpirVOpenGL || target == Target::SpirVVulkan)
        return convert_gxp_to_spirv(program, shader_name, features, target, hints, maskupdate, force_shader_debug, dumper);
    else
        return convert_gxp_to_glsl(program, shader_name, features, hints, maskupdate, force_shader_debug, dumper);
}

void convert_gxp_to_glsl_from_filepath(const std::string &shader_filepath) {
    const fs::path shader_filepath_str{ shader_filepath };
    std::ifstream gxp_stream(shader_filepath, std::ifstream::binary);

    if (!gxp_stream.is_open())
        return;

    const auto gxp_file_size = fs::file_size(shader_filepath_str);
    const auto gxp_program = static_cast<SceGxmProgram *>(calloc(gxp_file_size, 1));

    gxp_stream.read(reinterpret_cast<char *>(gxp_program), gxp_file_size);

    FeatureState features;
    features.direct_fragcolor = false;
    features.support_shader_interlock = true;

    Hints hints{
        .attributes = nullptr,
        .color_format = SCE_GXM_COLOR_FORMAT_U8U8U8U8_ABGR,
    };
    std::fill_n(hints.vertex_textures, SCE_GXM_MAX_TEXTURE_UNITS, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);
    std::fill_n(hints.fragment_textures, SCE_GXM_MAX_TEXTURE_UNITS, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR);

    convert_gxp(*gxp_program, shader_filepath_str.filename().string(), features, shader::Target::GLSLOpenGL, hints, false, true);

    free(gxp_program);
}

} // namespace shader
