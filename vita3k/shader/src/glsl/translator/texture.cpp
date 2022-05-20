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

#include <shader/disasm.h>
#include <shader/glsl/code_writer.h>
#include <shader/glsl/params.h>
#include <shader/glsl/translator.h>
#include <util/log.h>

using namespace shader;
using namespace usse;
using namespace glsl;

// Return the uv coefficients (a v4f32), each in [0,1]
// coords are the normalized coordinates in the sampled image
static std::string get_uv_coeffs(const std::string &sampled_image, const std::string &coords, std::string lod = "") {
    if (lod.empty()) {
        lod = std::string("textureQueryLod(") + sampled_image + ", " + coords + ").x";
    }

    // first we need to get the image size
    const std::string image_size = fmt::format("vec2(textureSize({}, {}))", sampled_image, lod);

    // un-normalize the coordinates, subtract 0.5 to each coord, the uv coefficients are the fractional values
    return std::string("fract(") + coords + " * " + image_size + " - 0.5)";
}

std::string USSETranslatorVisitorGLSL::do_fetch_texture(const std::string tex, std::string coord_name, const DataType dest_type, const int lod_mode,
    const std::string extra1, const std::string extra2, const int gather4_comp) {
    std::string result;

    switch (lod_mode) {
    case 4:
        result = fmt::format("textureProj({}, {})", tex, coord_name);
        break;

    case 5:
        result = fmt::format("textureProjCube({}, {})", tex, coord_name);
        break;

    case 0:
    case 1:
        result = fmt::format("texture({}, {})", tex, coord_name);
        break;

    case 2:
        result = fmt::format("textureLod({}, {}, {})", tex, coord_name, extra1);
        break;

    case 3:
        result = fmt::format("textureGrad({}, {}, {}, {})", tex, coord_name, extra1, extra2);
        break;

    default:
        return "";
    }

    if (is_integer_data_type(dest_type))
        result = variables.convert_to_int(result, 4, dest_type, true);

    return result;
}

void USSETranslatorVisitorGLSL::do_texture_queries(const NonDependentTextureQueryCallInfos &texture_queries) {
    Operand store_op;
    store_op.bank = RegisterBank::PRIMATTR;
    store_op.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;

    for (auto &texture_query : texture_queries) {
        store_op.type = static_cast<DataType>(texture_query.data_type);
        if (store_op.type == DataType::UNK) {
            // get the type from the hint
            store_op.type = texture_query.component_type;
        }

        bool proj = (texture_query.proj_pos >= 0);
        std::string coord_access = fmt::format("v_TexCoord{}", texture_query.coord_index);
        std::string proj_access = coord_access + ".";
        if (texture_query.sampler_cube) {
            coord_access += ".xyz";
        } else {
            coord_access += ".xy";
        }

        if (proj) {
            coord_access += static_cast<char>('w' + (texture_query.proj_pos + 1) % 4);
        }

        std::string fetch_result = do_fetch_texture(texture_query.sampler_name, coord_access, store_op.type, proj ? (texture_query.sampler_cube ? 5 : 4) : 0, "", "");
        store_op.num = texture_query.offset_in_pa;

        variables.store(store_op, fetch_result, 0b1111, 0, true);
    }
}

bool USSETranslatorVisitorGLSL::smp(
    ExtPredicate pred,
    Imm1 skipinv,
    Imm1 nosched,
    Imm1 syncstart,
    Imm1 minpack,
    Imm1 src0_ext,
    Imm1 src1_ext,
    Imm1 src2_ext,
    Imm2 fconv_type,
    Imm2 mask_count,
    Imm2 dim,
    Imm2 lod_mode,
    bool dest_use_pa,
    Imm2 sb_mode,
    Imm2 src0_type,
    Imm1 src0_bank,
    Imm2 drc_sel,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    if (!USSETranslatorVisitor::smp(pred, skipinv, nosched, syncstart, minpack, src0_ext, src2_ext, src2_ext, fconv_type,
            mask_count, dim, lod_mode, dest_use_pa, sb_mode, src0_type, src0_bank, drc_sel, src1_bank, src2_bank,
            dest_n, src0_n, src1_n, src2_n)) {
        return false;
    }

    // LOD mode: none, bias, replace, gradient
    if ((lod_mode != 0) && (lod_mode != 2) && (lod_mode != 3)) {
        LOG_ERROR("Sampler LOD replace not implemented!");
        return true;
    }

    // Base 0, turn it to base 1
    dim += 1;

    std::uint32_t coord_mask = 0b0011;
    if (dim == 3) {
        coord_mask = 0b0111;
    } else if (dim == 1) {
        coord_mask = 0b0001;
    }

    // Generate simple stuff
    // Load the coord
    std::string coords = variables.load(decoded_inst.opr.src0, coord_mask, 0);

    if (coords.empty()) {
        LOG_ERROR("Coord not loaded");
        return false;
    }

    if (dim == 1) {
        // It should be a line, so Y should be zero. There are only two dimensions texture, so this is a guess (seems concise)
        coords = fmt::format("vec2({}, 0)", coords);
        dim = 2;
    }

    bool is_texture_buffer_load = false;
    // only used for a load using the texture buffer
    std::string texture_index = "";
    if (!program_state.samplers.count(decoded_inst.opr.src1.num)) {
        if (program_state.texture_buffer_sa_offset == -1) {
            LOG_ERROR("Can't get the sampler (sampler doesn't exist!)");
            return true;
        }

        if (sb_mode != 0 || lod_mode != 2) {
            LOG_ERROR("Unhandled load using texture buffer with sb mode {} and lod mode {}", sb_mode, lod_mode);
            return true;
        }

        is_texture_buffer_load = true;
        decoded_inst.opr.src1.type = DataType::INT32;
        texture_index = variables.load(decoded_inst.opr.src1, 0b1, 0);
    }

    // if this is a texture buffer load, just attribute the first available sampler to it
    const SamplerInfo &sampler = is_texture_buffer_load ? program_state.samplers.begin()->second : program_state.samplers.at(decoded_inst.opr.src1.num);

    std::string result;
    if (sb_mode == 2) {
        if (lod_mode != 0)
            LOG_WARN("SMP info with non-zero lod mode is not implemented");

        // query info
        const std::string lod = std::string("textureQueryLod(") + sampler.name + ", " + coords + ").x";

        // xy are the uv coefficients
        std::string uv = get_uv_coeffs(sampler.name, coords, lod);
        // z is the trilinear fraction, w the LOD
        std::string tri_frac = std::string("fract(") + lod + ")";
        const std::string lod_level = std::string("uint(") + lod + ")";

        // the result is stored as a vector of uint8, we must convert it
        uv = variables.convert_to_int(uv, 2, DataType::UINT8, true);
        tri_frac = variables.convert_to_int(tri_frac, 1, DataType::UINT8, true);

        const std::string result = std::string("uvec4(") + uv + ", " + tri_frac + ", " + lod_level + ")";
        decoded_inst.opr.dest.type = DataType::UINT8;
        variables.store(decoded_inst.opr.dest, result, 0b1111, 0);
    } else {
        // Either LOD number or ddx
        std::string extra1;
        // ddy
        std::string extra2;

        if (lod_mode != 0) {
            switch (lod_mode) {
            case 2:
                extra1 = variables.load(decoded_inst.opr.src2, 0b1, 0);
                break;

            case 3:
                // Skip to next PA/INTERNAL to get DDY. Observed on shader in game like Hatsune Miku Diva X
                switch (dim) {
                case 2:
                    extra1 = variables.load(decoded_inst.opr.src2, 0b11, 0);
                    extra2 = variables.load(decoded_inst.opr.src2, 0b11, 2);
                    break;
                case 3:
                    extra1 = variables.load(decoded_inst.opr.src2, 0b111, 0);
                    extra2 = variables.load(decoded_inst.opr.src2, 0b111, 4);
                }

                break;

            default:
                break;
            }
        }

        if (is_texture_buffer_load) {
            // maybe put this in a function instead

            // do a big switch with all the different textures:
            // switch(texture_idx) {
            // case 0:
            //   dest = texture(texture0, pos);
            //   break;
            // case 1:
            //   dest = texture(texture1, pos);
            //   break;
            // ....

            std::vector<const SamplerInfo *> samplers;
            std::vector<int> sampler_indices;
            std::vector<int> index_to_segment;
            constexpr int sa_count = 32 * 4;
            // if dim is 2, do not look for cubes and if dim is 3, only look for cubes
            const bool request_cube = dim == 3;
            for (auto &smp : program_state.samplers) {
                if (smp.first < sa_count)
                    continue;

                if (request_cube != smp.second.is_cube)
                    continue;

                samplers.push_back(&smp.second);
                index_to_segment.push_back(sampler_indices.size());
                sampler_indices.push_back(smp.first - sa_count);
            }

            constexpr DataType tb_dest_fmt[] = {
                DataType::F32,
                DataType::UNK,
                DataType::F16,
                DataType::F32
            };

            if (!texture_index.empty()) {
                writer.add_to_current_body(std::string("switch (") + texture_index + ") {");
                for (size_t s = 0; s < samplers.size(); s++) {
                    const SamplerInfo *smp = samplers[s];

                    writer.add_to_current_body(fmt::format("case {}:", s));
                    writer.indent_current_body();
                    if (tb_dest_fmt[fconv_type] == DataType::UNK)
                        decoded_inst.opr.dest.type = smp->component_type;

                    std::string result = do_fetch_texture(smp->name, coords, decoded_inst.opr.dest.type, lod_mode, extra1);
                    const Imm4 dest_mask = (1U << smp->component_count) - 1;
                    variables.store(decoded_inst.opr.dest, result, dest_mask, 0);

                    writer.add_to_current_body("break;");
                    writer.dedent_current_body();
                }
                writer.add_to_current_body("}");
            } else if (samplers.size() > 0) {
                const SamplerInfo *smp = samplers[0];

                if (tb_dest_fmt[fconv_type] == DataType::UNK)
                    decoded_inst.opr.dest.type = smp->component_type;

                std::string result = do_fetch_texture(smp->name, coords, decoded_inst.opr.dest.type, lod_mode, extra1);
                const Imm4 dest_mask = (1U << smp->component_count) - 1;
                variables.store(decoded_inst.opr.dest, result, dest_mask, 0);
            } else {
                LOG_ERROR("Can't get the sampler (requested sampler doesn't exist!)");
            }
        } else if (sb_mode == 0) {
            std::string result = do_fetch_texture(sampler.name, coords, decoded_inst.opr.dest.type, lod_mode, extra1, extra2);
            const Imm4 dest_mask = (1U << sampler.component_count) - 1;
            variables.store(decoded_inst.opr.dest, result, dest_mask, 0);
        } else {
            // sb_mode = 1 or 3 : gather 4 (+ uv if sb_mode = 3)
            // first gather all components
            std::vector<std::string> g4_comps;
            for (int comp = 0; comp < sampler.component_count; comp++) {
                g4_comps.push_back(do_fetch_texture(sampler.name, coords, decoded_inst.opr.dest.type, lod_mode, extra1, extra2, comp));
            }

            if (sampler.component_count == 1) {
                // easy, no need to do all this reordering
                variables.store(decoded_inst.opr.dest, g4_comps[0], 0b1111, 0);
                decoded_inst.opr.dest.num += get_data_type_size(decoded_inst.opr.dest.type);
            } else {
                if (sampler.component_count == 3)
                    LOG_ERROR("Sampler is not supposed to have 3 components");

                // we have the values in the following layout x1 x2 ... y1 y2 ...
                // we want to store them with the layout x1 x2 x3 ... y1 y2 y3 ...
                std::vector<std::string> comps_alone;
                for (int pixel = 0; pixel < 4; pixel++) {
                    for (int comp = 0; comp < sampler.component_count; comp++) {
                        comps_alone.push_back(g4_comps[comp] + "." + std::string(1, static_cast<char>('w' + (pixel + 1) % 4)));
                    }
                }

                for (size_t idx = 0; idx < comps_alone.size(); idx += 4) {
                    // pack them by 4 so each pack size is a multiple of 32 bitsv
                    const std::string comp_packed = fmt::format("{}({}, {}, {}, {})",
                        is_unsigned_integer_data_type(decoded_inst.opr.dest.type) ? "uvec" : (is_signed_integer_data_type(decoded_inst.opr.dest.type) ? "ivec" : "vec"),
                        comps_alone[idx], comps_alone[idx + 1], comps_alone[idx + 2], comps_alone[idx + 3]);                    
                    variables.store(decoded_inst.opr.dest, comp_packed, 0b1111, 0);

                    decoded_inst.opr.dest.num += get_data_type_size(decoded_inst.opr.dest.type);
                }
            }

            if (sb_mode == 3) {
                // compute and save bilinear coefficients
                const std::string uv = get_uv_coeffs(sampler.name, coords);
                // the pixels returned by gather4 are in the following order : (0,1) (1,1) (1,0) (0,0)
                // so at the end we want (1-u)v uv u(1-v) (1-u)(1-v)
                // however, looking at the generated shader code in some games, it looks like the coefficients are
                // expected to be in this order but reversed...
                const std::string u = uv + ".x";
                const std::string v = uv + ".y";

                const std::string onemu = std::string("1.0 - ") + u;
                const std::string onemv = std::string("1.0 - ") + v;

                // (1-u) u
                const std::string x_coeffs = std::string("vec2(") + onemu + ", " + u + ")";
                // (1-u)v uv
                const std::string comp1 = x_coeffs + " * " + v;
                // (1-u)(1-v) u(1-v)
                const std::string comp2 = x_coeffs + " * " + onemv;
                // (1-u)v uv u(1-v) (1-u)(1-v) in reversed order
                const std::string coeffs = std::string("vec4(") + comp1 + ", " + comp2 + ").zwyx";

                // bilinear coeffs are stored as float16
                decoded_inst.opr.dest.type = DataType::F16;
                variables.store(decoded_inst.opr.dest, coeffs, 0b1111, 0);
            }
        }
        result = do_fetch_texture(sampler.name, coords, DataType::F32, lod_mode, extra1, extra2);

        const Imm4 dest_mask = (1U << sampler.component_count) - 1;
        variables.store(decoded_inst.opr.dest, result, dest_mask, 0);
    }


    return true;
}
