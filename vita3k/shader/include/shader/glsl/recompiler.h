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

#include <gxm/types.h>
#include <shader/recompiler.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

struct SceGxmProgram;

namespace shader::usse::glsl {
struct SamplerInfo {
    std::string name;
    bool is_cube;
    DataType component_type;
    uint8_t component_count;

    SamplerInfo()
        : name()
        , is_cube()
        , component_type()
        , component_count() {
    }
    SamplerInfo(const std::string &name, const bool is_cube,
        const DataType component_type = DataType::F32, const uint8_t component_count = 4)
        : name(name)
        , is_cube(is_cube)
        , component_type(component_type)
        , component_count(component_count) {
    }
    ~SamplerInfo() {
    }
};

using SamplerMap = std::map<uint32_t, SamplerInfo>;

struct VarToReg {
    std::string var_name;
    int offset;
    int location;
    int comp_count;
    DataType data_type;
    bool reg_format;
    bool builtin;
};

struct NonDependentTextureQueryCallInfo {
    int sampler_index;
    int coord_index;
    int proj_pos;
    std::string sampler_name;
    bool sampler_cube;

    int data_type;
    int offset_in_pa;

    DataType component_type;
};

using NonDependentTextureQueryCallInfos = std::vector<NonDependentTextureQueryCallInfo>;

struct ProgramState {
    // Sampler map. Since all banks are a flat array, sampler must be in an explicit bank.
    SamplerMap samplers;

    // when using a thread, texture or literal buffer, if not -1, this fields contain the sa register
    // with the matching address, this assumes of course that this address is not copied somewhere
    // else and that this register is not overwritten
    // the base field is the offset to be applied when reading this buffer (almost always -4)
    int texture_buffer_sa_offset;
    int texture_buffer_base;

    int literal_buffer_sa_offset;
    int literal_buffer_base;

    int thread_buffer_sa_offset;
    int thread_buffer_base;

    NonDependentTextureQueryCallInfos non_dependent_queries;
    const SceGxmProgram &actual_program;
    bool should_generate_vld_func;

    explicit ProgramState(const SceGxmProgram &actual_program)
        : texture_buffer_sa_offset(-1)
        , texture_buffer_base(0)
        , literal_buffer_sa_offset(-1)
        , literal_buffer_base(0)
        , thread_buffer_sa_offset(-1)
        , thread_buffer_base(0)
        , actual_program(actual_program)
        , should_generate_vld_func(false) {
    }
};
} // namespace shader::usse::glsl

namespace shader {
GeneratedShader convert_gxp_to_glsl(const SceGxmProgram &program, const std::string &shader_hash, const FeatureState &features, const Hints &hints, bool maskupdate,
    bool force_shader_debug, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper);
} // namespace shader
