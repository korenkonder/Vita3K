// Vita3K emulator project
// Copyright (C) 2024 Vita3K team
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

#pragma once

#include <shader/recompiler.h>
#include <shader/translator_types.h>

#include <string>
#include <vector>

struct SceGxmProgram;

namespace shader {
// Dump generated SPIR-V disassembly up to this point
void spirv_disasm_print(const usse::SpirvCode &spirv_binary, std::string *spirv_dump = nullptr);

GeneratedShader convert_gxp_to_spirv(const SceGxmProgram &program, const std::string &shader_hash, const FeatureState &features, const Target target, const Hints &hints, bool maskupdate,
    bool force_shader_debug, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper);
} // namespace shader
