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

#include <features/state.h>
#include <gxm/functions.h>
#include <gxm/types.h>
#include <shader/disasm.h>
#include <shader/glsl/code_writer.h>
#include <shader/glsl/consts.h>
#include <shader/glsl/params.h>
#include <shader/glsl/recompiler.h>
#include <shader/glsl/translator.h>
#include <shader/gxp_parser.h>
#include <shader/types.h>

#include <util/bit_cast.h>
#include <util/log.h>
#include <util/overloaded.h>

namespace shader {
// **************
// * Constants *
// **************

static constexpr int REG_PA_COUNT = 32 * 4;
static constexpr int REG_SA_COUNT = 32 * 4;
static constexpr int REG_I_COUNT = 3 * 4;
static constexpr int REG_TEMP_COUNT = 20 * 4;
static constexpr int REG_INDEX_COUNT = 2 * 4;
static constexpr int REG_PRED_COUNT = 4 * 4;
static constexpr int REG_O_COUNT = 20 * 4;

struct TranslationState {
    std::string image_storage_format;
};
} // namespace shader

namespace shader::usse::glsl {
static void nicen_name_for_glsl_rules(std::string &prev) {
    while (true) {
        std::size_t dot_pos = prev.find('.');
        if (dot_pos != std::string::npos) {
            prev.replace(dot_pos, 1, "_");
        } else {
            break;
        }
    }

    while (true) {
        std::size_t bracket_open_pos = prev.find('[');
        if (bracket_open_pos != std::string::npos) {
            prev.replace(bracket_open_pos, 1, "_");
        } else {
            break;
        }
    }

    while (true) {
        std::size_t bracket_open_pos = prev.find(']');
        if (bracket_open_pos != std::string::npos) {
            prev.replace(bracket_open_pos, 1, "_");
        } else {
            break;
        }
    }

    // GLSL does not allow name with double dash inside
    while (true) {
        std::size_t pos_double_dash = prev.find("__");
        if (pos_double_dash != std::string::npos) {
            prev.replace(pos_double_dash, 2, "_dd_");
        } else {
            break;
        }
    }
}
static void create_uniform_buffers(CodeWriter &writer, const SceGxmProgram &program, const ProgramInput &input, std::map<int, int> &buffer_bases) {
    std::map<int, std::uint32_t> buffer_sizes;

    for (const auto &buffer : input.uniform_buffers) {
        if (buffer.index >= SCE_GXM_REAL_MAX_UNIFORM_BUFFER)
            continue;

        const auto buffer_size = (buffer.size + 3) / 4;
        buffer_sizes.emplace(buffer.index, buffer_size);
    }

    if (buffer_sizes.empty()) {
        return;
    }

    writer.add_declaration(fmt::format("layout (std140, binding = {}) buffer {}Type {{", program.is_vertex() ? 0 : 1, program.is_vertex() ? VERTEX_UB_GROUP_NAME : FRAGMENT_UB_GROUP_NAME));
    writer.indent_declaration();
    std::uint32_t last_offset = 0;
    for (auto [buffer_index, buffer_size] : buffer_sizes) {
        writer.add_declaration(fmt::format("vec4 buffer{}[{}];", buffer_index, buffer_size));
        buffer_bases.emplace(buffer_index, last_offset);

        last_offset += buffer_size * 16;
    }
    writer.dedent_declaration();

    writer.add_declaration(fmt::format("}} {};\n", program.is_vertex() ? VERTEX_UB_GROUP_NAME : FRAGMENT_UB_GROUP_NAME));
}

static void create_samplers(CodeWriter &writer, SamplerMap &sampler_map, const SceGxmProgram &program, const ProgramInput &input, const Hints *hints = nullptr) {
    for (const auto &sampler : input.samplers) {
        std::string sampler_name = fmt::format("{}_{}", program.is_vertex() ? "vertTex" : "fragTex", sampler.name);
        nicen_name_for_glsl_rules(sampler_name);

        writer.add_declaration(fmt::format("layout (binding = {}) uniform {} {};", sampler.index + (program.is_vertex() ? SCE_GXM_MAX_TEXTURE_UNITS : 0),
            sampler.is_cube ? "samplerCube" : "sampler2D", sampler_name));

        const SceGxmTextureFormat texture_format = program.is_fragment() ? hints->fragment_textures[sampler.index] : hints->vertex_textures[sampler.index];
        sampler_map[sampler.index] = SamplerInfo(sampler_name,
            sampler.is_cube,
            get_texture_component_type(texture_format),
            get_texture_component_count(texture_format));
    }
}

static std::string create_builtin_sampler(CodeWriter &writer, const FeatureState &features, TranslationState &translation_state, const int binding, const std::string &name) {
    std::string format = translation_state.image_storage_format;
    if (name == "f_mask")
        // f_mask is always rgba8
        format = "rgba8";

    if (format.empty())
        return fmt::format("layout (binding = {}) uniform image2D {};", binding, name);
    else
        return fmt::format("layout (binding = {}, {}) uniform image2D {};", binding, format, name);
}

static void create_fragment_inputs(CodeWriter &writer, ShaderVariables &variables, const FeatureState &features, TranslationState &translation_state, const SceGxmProgram &program) {
    writer.add_declaration(create_builtin_sampler(writer, features, translation_state, MASK_TEXTURE_SLOT_IMAGE, "f_mask"));
    writer.add_to_preload("if (all(lessThan(imageLoad(f_mask, ivec2(gl_FragCoord.xy / float(renderFragInfo.res_multiplier))), vec4(0.5)))) discard;");

    if (program.is_frag_color_used()) {
        if (features.direct_fragcolor) {
            // The GPU supports gl_LastFragData. It's only OpenGL though
            if (features.preserve_f16_nan_as_u16) {
                writer.add_to_preload("if (renderFragInfo.use_raw_image >= 0.5) {");
                writer.indent_preload();
                writer.add_to_preload("o0 = gl_LastFragData[1].xyzw;");
                writer.dedent_preload();
                writer.add_to_preload("} else {");
                writer.indent_preload();
                writer.add_to_preload("o0 = gl_LastFragData[0];");
                writer.dedent_preload();
                writer.add_to_preload("}");
            } else {
                writer.add_to_preload("o0 = gl_LastFragData[0];");
            }
        } else if (features.support_shader_interlock || features.support_texture_barrier) {
            writer.add_declaration(create_builtin_sampler(writer, features, translation_state, COLOR_ATTACHMENT_TEXTURE_SLOT_IMAGE, "f_colorAttachment"));
            writer.add_declaration("");

            variables.should_gen_pack_unpack[ShaderVariables::GEN_PACK_4XU8] = true;
            variables.should_gen_pack_unpack[ShaderVariables::GEN_PACK_2XU16] = true;

            if (features.preserve_f16_nan_as_u16) {
                writer.add_declaration(fmt::format("layout (binding = {}, rgba16ui) uniform uimage2D f_colorAttachment_rawUI;", COLOR_ATTACHMENT_RAW_TEXTURE_SLOT_IMAGE));
                writer.add_to_preload("if (renderFragInfo.use_raw_image >= 0.5) {");
                writer.indent_preload();
                writer.add_to_preload("o0.xy = pack4xU16(imageLoad(f_colorAttachment_rawUI, ivec2(gl_FragCoord.xy)));");
                writer.dedent_preload();
                writer.add_to_preload("} else {");
                writer.indent_preload();
                writer.add_to_preload("o0.x = pack4xU8(uvec4(imageLoad(f_colorAttachment, ivec2(gl_FragCoord.xy)) * vec4(255.0)));");
                writer.dedent_preload();
                writer.add_to_preload("}");
            } else {
                writer.add_to_preload("o0.x = pack4xU8(uvec4(imageLoad(f_colorAttachment, ivec2(gl_FragCoord.xy)) * vec4(255.0)));");
            }

            variables.mark_f32_raw_dirty(RegisterBank::OUTPUT, 0);
            variables.mark_f32_raw_dirty(RegisterBank::OUTPUT, 1);
        } else {
            // Try to initialize outs[0] to some nice value. In case the GPU has garbage data for our shader
            writer.add_to_preload("o0 = vec4(0.0);");
        }
    }
}

static void create_parameters(ProgramState &state, CodeWriter &writer, ShaderVariables &params, const FeatureState &features,
    TranslationState &translation_state, const SceGxmProgram &program, const ProgramInput &program_input,
    const Hints *hints = nullptr) {
    if (program.is_fragment()) {
        writer.add_to_preload("if ((renderFragInfo.front_disabled != 0.0) && gl_FrontFacing) discard;");
        writer.add_to_preload("if ((renderFragInfo.back_disabled != 0.0) && !gl_FrontFacing) discard;");
    }

    writer.add_declaration("bool p0;");
    writer.add_declaration("bool p1;");
    writer.add_declaration("bool p2;");
    writer.add_declaration("float temp1;");
    writer.add_declaration("vec2 temp2;");
    writer.add_declaration("vec3 temp3;");
    writer.add_declaration("vec4 temp4;");
    writer.add_declaration("float temp1_1;");
    writer.add_declaration("vec2 temp2_1;");
    writer.add_declaration("vec3 temp3_1;");
    writer.add_declaration("vec4 temp4_1;");
    writer.add_declaration("float temp1_2;");
    writer.add_declaration("vec2 temp2_2;");
    writer.add_declaration("vec3 temp3_2;");
    writer.add_declaration("vec4 temp4_2;");
    writer.add_declaration("int itemp1;");
    writer.add_declaration("ivec2 itemp2;");
    writer.add_declaration("ivec3 itemp3;");
    writer.add_declaration("ivec4 itemp4;");
    writer.add_declaration("int base_temp;");
    writer.add_declaration("\n");

    std::map<int, int> buffer_bases;
    SamplerMap samplers;
    std::vector<VarToReg> var_to_regs;

    create_uniform_buffers(writer, program, program_input, buffer_bases);
    create_samplers(writer, samplers, program, program_input, hints);

    // create thread buffer if it is being used
    if (program.thread_buffer_count > 0) {
        // there are 4 cores, each having 4 pipelines, assume the thread buffer is evenly divided
        // between all of them and each pipeline only does things in its segment
        const uint32_t size_in_f32 = program.thread_buffer_count / (4 * 4 * sizeof(float));
        writer.add_declaration(fmt::format("float thread_buffer[{}];", size_in_f32));
    }

    int32_t in_fcount_allocated = 0;
    const bool has_texture_buffer = program.texture_buffer_count > 0;

    for (const auto &input : program_input.inputs) {
        std::visit(overloaded{
                       [&](const LiteralInputSource &s) {
                           params.mark_f32_raw_dirty(RegisterBank::SECATTR, input.offset);
                           writer.add_to_current_body(fmt::format("sa{}.{} = uintBitsToFloat(0x{:X});", input.offset / 4 * 4, static_cast<char>('w' + ((input.offset + 1) % 4)),
                               std::bit_cast<std::uint32_t>(s.data)));
                       },
                       [&](const UniformBufferInputSource &s) {
                           if (s.index == SCE_GXM_TEXTURE_BUFFER) {
                               state.texture_buffer_sa_offset = input.offset;
                               state.texture_buffer_base = s.base;
                           } else if (s.index == SCE_GXM_LITERAL_BUFFER) {
                               state.literal_buffer_sa_offset = input.offset;
                               state.literal_buffer_base = s.base;
                           } else if (s.index == SCE_GXM_THREAD_BUFFER) {
                               state.thread_buffer_sa_offset = input.offset;
                               state.thread_buffer_base = s.base;
                           } else {
                               const int base = buffer_bases.at(s.index) + s.base;
                               params.mark_f32_raw_dirty(RegisterBank::SECATTR, input.offset);
                               writer.add_to_current_body(fmt::format("sa{}.{} = intBitsToFloat({});", static_cast<std::int32_t>(input.offset) / 4 * 4, static_cast<char>('w' + ((input.offset + 1) % 4)), base));
                           }
                       },
                       [&](const DependentSamplerInputSource &s) {
                           const auto sampler = samplers.at(s.index);
                           state.samplers.emplace(input.offset, sampler);

                           if (has_texture_buffer && s.layout_position == 0) {
                               // store the index in the first texture register to track it
                               Operand reg;
                               reg.bank = RegisterBank::SECATTR;
                               reg.num = input.offset;
                               reg.type = DataType::INT32;
                               params.store(reg, fmt::format("{}", s.index), 0b1, 0);
                           }
                       },
                       [&](const AttributeInputSource &s) {
                           if (input.bank == RegisterBank::SPECIAL) {
                               // Texcoord base location
                               writer.add_declaration(fmt::format("layout (location = {}) in vec4 v_TexCoord{};", s.opt_location + 4, s.index));
                           } else {
                               const int type_size = get_data_type_size(input.type);
                               DataType dt = input.type;
                               VarToReg var_to_reg;
                               if (s.regformat) {
                                   if (type_size == 1)
                                       dt = DataType::UINT8;
                                   else if (type_size == 2)
                                       dt = DataType::UINT16;
                                   else
                                       dt = DataType::UINT32;

                                   int num_comp = input.array_size * input.component_count;
                                   if (input.type == DataType::C10)
                                       // this is 10-bit and not 8-bit
                                       num_comp = (num_comp * 10 + 7) / 8;

                                   var_to_reg.comp_count = num_comp;
                               } else {
                                   var_to_reg.comp_count = input.array_size * input.component_count;
                               }

                               var_to_reg.offset = input.offset;
                               var_to_reg.reg_format = s.regformat;
                               var_to_reg.var_name = s.name;
                               var_to_reg.location = (s.opt_location != 0xFFFFFFFF) ? s.opt_location : in_fcount_allocated / 4;
                               var_to_reg.builtin = false;


                               // Some compilers does not allow in variables to be casted
                               switch (s.semantic) {
                               case SCE_GXM_PARAMETER_SEMANTIC_INDEX:
                                   writer.add_to_preload("int vertexIdTemp = gl_VertexID;");
                                   var_to_reg.var_name = "intBitsToFloat(vertexIdTemp)";
                                   var_to_reg.data_type = DataType::F32;
                                   var_to_reg.comp_count = 1;
                                   var_to_reg.builtin = true;

                                   break;

                               case SCE_GXM_PARAMETER_SEMANTIC_INSTANCE:
                                   writer.add_to_preload("int instanceIdTemp = gl_InstanceID;");
                                   var_to_reg.var_name = "intBitsToFloat(instanceIdTemp)";
                                   var_to_reg.data_type = DataType::F32;
                                   var_to_reg.comp_count = 1;
                                   var_to_reg.builtin = true;

                                   break;

                               default:
                                   var_to_reg.data_type = dt;
                                   break;
                               }

                               var_to_regs.push_back(var_to_reg);
                               in_fcount_allocated += ((input.array_size * input.component_count + 3) / 4 * 4);
                           }
                       },
                       [&](const NonDependentSamplerSampleSource &s) {
                           const SamplerInfo &sampler = samplers[s.sampler_index];

                           NonDependentTextureQueryCallInfo info;
                           info.sampler_name = sampler.name;
                           info.sampler_cube = sampler.is_cube;
                           info.sampler_index = s.sampler_index;
                           info.coord_index = s.coord_index;
                           info.proj_pos = s.proj_pos;
                           info.data_type = static_cast<int>(input.type);
                           info.offset_in_pa = input.offset;
                           info.component_type = sampler.component_type;

                           state.non_dependent_queries.push_back(info);
                       } },
            input.source);
    }

    if (has_texture_buffer) {
        // we need to get an access to all the samplers
        // easiest way it to store them in the sampler map, sa register go up to 128, so store them after this limit
        for (auto &sampler : samplers)
            state.samplers.emplace(REG_SA_COUNT + sampler.first, sampler.second);
    }

    if (program.is_vertex() && (in_fcount_allocated == 0) && (program.primary_reg_count != 0)) {
        // Using hint to create attribute. Looks like attribute with F32 types are stripped, otherwise
        // whole shader symbols are kept...
        if (hints) {
            LOG_INFO("Shader stripped all symbols, trying to use hint attributes");

            const auto &attributes = hints->attributes;
            for (std::size_t i = 0; i < attributes->size(); i++) {
                VarToReg var_to_reg;
                var_to_reg.location = attributes->at(i).regIndex / 4;
                var_to_reg.offset = attributes->at(i).regIndex - var_to_reg.location * 4;
                var_to_reg.comp_count = (attributes->at(i).componentCount + 3) / 4 * 4;
                var_to_reg.data_type = DataType::F32;
                var_to_reg.reg_format = false;
                var_to_reg.builtin = false;
                var_to_reg.var_name = fmt::format("attribute{}", i);

                var_to_regs.push_back(var_to_reg);
            }
        }
    }

    // Store var to regs
    for (auto &var_to_reg : var_to_regs) {
        nicen_name_for_glsl_rules(var_to_reg.var_name);

        if (!var_to_reg.builtin) {
            if (var_to_reg.reg_format && is_integer_data_type(var_to_reg.data_type)) {
                writer.add_declaration(fmt::format("layout (location = {}) in {}vec4 {};", var_to_reg.location,
                    (is_signed_integer_data_type(var_to_reg.data_type) ? "i" : "u"), var_to_reg.var_name));
            } else {
                writer.add_declaration(fmt::format("layout (location = {}) in vec4 {};", var_to_reg.location, var_to_reg.var_name));
            }
        }

        Operand dest;
        dest.num = var_to_reg.offset;
        dest.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        dest.bank = RegisterBank::PRIMATTR;
        dest.type = var_to_reg.data_type;

        bool is_4th_component_1 = false;
        if (!var_to_reg.builtin && !features.support_rgb_attributes && program.is_vertex() && var_to_reg.comp_count == 4) {
            // if the vertex input was rgb, the alpha component must be set to 1,
            // however it will be set to whatever is in memory after the blue component
            for (const auto &attribute : *hints->attributes) {
                if (attribute.regIndex == var_to_reg.offset) {
                    is_4th_component_1 = (attribute.componentCount == 3);
                    break;
                }
            }
        }

        if (is_4th_component_1) {
            // set the 4th component to 1, because it's what the shader is expecting it to be
            var_to_reg.var_name = fmt::format("vec4({}.xyz, 1.0)", var_to_reg.var_name);
        }

        if (is_integer_data_type(dest.type) && !var_to_reg.reg_format) {
            var_to_reg.var_name = params.convert_to_int(var_to_reg.var_name, 4, dest.type, true);
        }

        if (var_to_reg.comp_count == 4) {
            params.store(dest, var_to_reg.var_name, 0b1111, 0, true);
        } else {
            std::string swizz_str = std::string(".xyzw").substr(0, var_to_reg.comp_count + 1);
            params.store(dest, var_to_reg.var_name + swizz_str, 0b1111 >> (4 - var_to_reg.comp_count), 0, true);
        }
    }

    if (program.is_vertex()) {
        // Add vertex render info
        writer.add_declaration("layout (std140, binding = 2) uniform GxmRenderVertBufferBlock {");
        writer.indent_declaration();
        writer.add_declaration("vec4 viewport_flip;");
        writer.add_declaration("float viewport_flag;");
        writer.add_declaration("float screen_width;");
        writer.add_declaration("float screen_height;");
        writer.add_declaration("float z_offset;");
        writer.add_declaration("float z_scale;");
        writer.dedent_declaration();
        writer.add_declaration("} renderVertInfo;");
    } else {
        // Add fragment render info
        writer.add_declaration("layout (std140, binding = 3) uniform GxmRenderFragBufferBlock {");
        writer.indent_declaration();
        writer.add_declaration("float back_disabled;");
        writer.add_declaration("float front_disabled;");
        writer.add_declaration("float writing_mask;");
        writer.add_declaration("float use_raw_image;");
        writer.add_declaration("int res_multiplier;");
        writer.dedent_declaration();
        writer.add_declaration("} renderFragInfo;");

        create_fragment_inputs(writer, params, features, translation_state, program);
    }

    writer.add_declaration("\n");
}

struct VertexProgramOutputProperties {
    std::string name;
    std::uint32_t component_count;
    std::uint32_t location;

    VertexProgramOutputProperties()
        : name(nullptr)
        , component_count(0)
        , location(0) {}

    VertexProgramOutputProperties(const char *name, std::uint32_t component_count, std::uint32_t location)
        : name(name)
        , component_count(component_count)
        , location(location) {}
};

using VertexProgramOutputPropertiesMap = std::map<SceGxmVertexProgramOutputs, VertexProgramOutputProperties>;

static void create_output(ProgramState &state, CodeWriter &writer, ShaderVariables &params, const FeatureState &features, const SceGxmProgram &program, const Hints *hints) {
    if (program.is_vertex()) {
        gxp::GxmVertexOutputTexCoordInfos coord_infos;
        SceGxmVertexProgramOutputs vertex_outputs = gxp::get_vertex_outputs(program, &coord_infos);

        static const auto calculate_copy_comp_count = [](uint8_t info) {
            // TexCoord info uses preset values described below for determining lengths.
            uint8_t length = 0;
            if (info & 0b001u)
                length += 2; // uses xy
            if (info & 0b010u)
                length += 1; // uses z
            if (info & 0b100u)
                length += 1; // uses w

            return length;
        };

        VertexProgramOutputPropertiesMap vertex_properties_map;

        // list is used here to gurantee the vertex outputs are written in right order
        std::list<SceGxmVertexProgramOutputs> vertex_outputs_list;
        const auto add_vertex_output_info = [&](SceGxmVertexProgramOutputs vo, const char *name, std::uint32_t component_count, std::uint32_t location) {
            vertex_properties_map.emplace(vo, VertexProgramOutputProperties(name, component_count, location));
            vertex_outputs_list.push_back(vo);
        };

        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_POSITION, "v_Position", 4, 0);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_COLOR0, "v_Color0", 4, 1);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_COLOR1, "v_Color1", 4, 2);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_FOG, "v_Fog", 2, 3);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD0, "v_TexCoord0", calculate_copy_comp_count(coord_infos[0]), 4);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD1, "v_TexCoord1", calculate_copy_comp_count(coord_infos[1]), 5);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD2, "v_TexCoord2", calculate_copy_comp_count(coord_infos[2]), 6);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD3, "v_TexCoord3", calculate_copy_comp_count(coord_infos[3]), 7);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD4, "v_TexCoord4", calculate_copy_comp_count(coord_infos[4]), 8);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD5, "v_TexCoord5", calculate_copy_comp_count(coord_infos[5]), 9);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD6, "v_TexCoord6", calculate_copy_comp_count(coord_infos[6]), 10);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD7, "v_TexCoord7", calculate_copy_comp_count(coord_infos[7]), 11);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD8, "v_TexCoord8", calculate_copy_comp_count(coord_infos[8]), 12);
        add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_TEXCOORD9, "v_TexCoord9", calculate_copy_comp_count(coord_infos[9]), 13);
        // TODO: this should be translated to gl_PointSize
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_PSIZE, "v_Psize", 1);
        // TODO: these should be translated to gl_ClipDistance
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP0, "v_Clip0", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP1, "v_Clip1", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP2, "v_Clip2", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP3, "v_Clip3", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP4, "v_Clip4", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP5, "v_Clip5", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP6, "v_Clip6", 1);
        // add_vertex_output_info(SCE_GXM_VERTEX_PROGRAM_OUTPUT_CLIP7, "v_Clip7", 1);

        Operand o_op;
        o_op.bank = RegisterBank::OUTPUT;
        o_op.num = 0;
        o_op.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        o_op.type = DataType::F32;

        for (const auto vo : vertex_outputs_list) {
            if (vertex_outputs & vo) {
                const auto vo_typed = static_cast<SceGxmVertexProgramOutputs>(vo);
                VertexProgramOutputProperties properties = vertex_properties_map.at(vo_typed);

                // Do store
                const Imm4 load_mask = (vo == SCE_GXM_VERTEX_PROGRAM_OUTPUT_PSIZE) ? 0b1 : 0b1111;
                std::string o_val = params.load(o_op, load_mask, 0);

                if (vo == SCE_GXM_VERTEX_PROGRAM_OUTPUT_POSITION) {
                    writer.add_to_current_body(fmt::format("gl_Position = {} * renderVertInfo.viewport_flip;", o_val));
                    // Transform screen space coordinate to ndc when viewport is disabled.
                    writer.add_to_current_body("if (renderVertInfo.viewport_flag < 0.5) {");
                    writer.indent_current_body();
                    // o_val2 = (x,y,z,w) * (2/width, -2/height, 1, 1) + (-1,1,0,0)
                    writer.add_to_current_body("gl_Position.xy = gl_Position.xy * vec2(2.0 / renderVertInfo.screen_width, -2.0 / renderVertInfo.screen_height) + vec2(-1.0, 1.0);");
                    // Note: Depth range and user clip planes are ineffective in this mode
                    // However, that can't be directly translated, so we just gonna set it to w here
                    writer.add_to_current_body("gl_Position.z = gl_Position.w;");
                    writer.dedent_current_body();
                    writer.add_to_current_body("} else {");
                    writer.indent_current_body();
                    // scale the depth and w coordinate
                    // screen_z = z_offset + z_scale * (z / w)
                    // convert [0,1] depth range (gxp, vulkan) to [-1,1] depth range (opengl)
                    // multiply by w because it will be re-divided by w during screen normalization
                    writer.add_to_current_body("gl_Position.z = ((renderVertInfo.z_offset + renderVertInfo.z_scale * (gl_Position.z / gl_Position.w)) * 2.0 - 1.0) * gl_Position.w;");
                    writer.dedent_current_body();

                    writer.add_to_current_body("}");
                } else if (vo == SCE_GXM_VERTEX_PROGRAM_OUTPUT_PSIZE) {
                    writer.add_to_current_body(fmt::format("gl_PointSize = {};", o_val));
                } else {
                    writer.add_declaration(fmt::format("layout (location = {}) out vec4 {};", properties.location, properties.name));
                    writer.add_to_current_body(fmt::format("{} = {};", properties.name, o_val));
                }

                o_op.num += properties.component_count;
            }
        }
    } else {
        auto vertex_varyings_ptr = program.vertex_varyings();

        Operand color_val_operand;
        color_val_operand.bank = program.is_native_color() ? RegisterBank::OUTPUT : RegisterBank::PRIMATTR;
        color_val_operand.num = 0;
        color_val_operand.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        color_val_operand.type = std::get<0>(shader::get_parameter_type_store_and_name(program.get_fragment_output_type()));

        // if the output component count is greater than the surface component count,
        // it means we must pack multiple components (with lower precision) into one of the surface component
        // this is used in assassin creed 3
        if (gxm::get_base_format(hints->color_format) == SCE_GXM_COLOR_BASE_FORMAT_F32F32 && vertex_varyings_ptr->output_comp_count > 2) {
            if (color_val_operand.type == DataType::F16)
                color_val_operand.type = DataType::F32;
        }

        if (!program.is_native_color() && vertex_varyings_ptr->output_param_type == 1) {
            color_val_operand.num = vertex_varyings_ptr->fragment_output_start;
        }

        std::string result = params.load(color_val_operand, 0b1111, 0);
        if (is_unsigned_integer_data_type(color_val_operand.type)) {
            result = fmt::format("vec4(uvec4({})) / 255.0", result);
        }

        bool use_outs = true;

        if (program.is_frag_color_used()) {
            if (features.is_programmable_blending_need_to_bind_color_attachment()) {
                writer.add_to_current_body(fmt::format("imageStore(f_colorAttachment, ivec2(gl_FragCoord.xy), {});", result));

                if (features.preserve_f16_nan_as_u16) {
                    color_val_operand.type = DataType::UINT16;
                    result = params.load(color_val_operand, 0b1111, 0);

                    writer.add_to_current_body(fmt::format("imageStore(f_colorAttachment_rawUI, ivec2(gl_FragCoord.xy), {});", result));
                }

                use_outs = false;
            }
        }

        if (use_outs) {
            writer.add_declaration("layout (location = 0) out vec4 out_color;");
            writer.add_to_current_body(fmt::format("out_color = {};", result));

            if (features.preserve_f16_nan_as_u16) {
                color_val_operand.type = DataType::UINT16;
                result = params.load(color_val_operand, 0b1111, 0);

                writer.add_declaration("layout (location = 1) out uvec4 out_color_ui;");
                writer.add_to_current_body(fmt::format("out_color_ui = {};", result));
            }
        }
    }

    writer.add_declaration("\n");
}

static void create_necessary_headers(CodeWriter &writer, const SceGxmProgram &program, const FeatureState &features, const std::string &shader_name) {
    // Must at least be version 430
    writer.add_declaration("#version 430");
    writer.add_declaration("");
    writer.add_declaration(std::string("// Shader Name: ") + shader_name);

    if (program.is_fragment() && program.is_frag_color_used()) {
        if (features.direct_fragcolor) {
            writer.add_declaration("#extension GL_EXT_shader_framebuffer_fetch: require");
        } else if (features.should_use_shader_interlock()) {
            writer.add_declaration("#extension GL_ARB_fragment_shader_interlock: require");
            writer.add_to_preload("beginInvocationInterlockARB();");
        }

        if (features.support_unknown_format) {
            writer.add_declaration("#extension GL_EXT_shader_image_load_formatted: require");
        }
    }

    writer.add_declaration("");
    if (program.is_fragment() && program.is_frag_color_used() && !features.direct_fragcolor) {
        writer.add_declaration("layout(early_fragment_tests) in;");
    }
}

static void create_necessary_footers(CodeWriter &writer, const SceGxmProgram &program, const FeatureState &features) {
    if (program.is_fragment() && program.is_frag_color_used()) {
        if (features.should_use_shader_interlock()) {
            writer.add_to_current_body("endInvocationInterlockARB();");
        }
    }
}

static void create_program_needed_functions(const ProgramState &state, const ProgramInput &input, CodeWriter &writer) {
    if (state.should_generate_vld_func) {
        writer.add_declaration("float fetchMemory(int offset) {");
        writer.indent_declaration();

        const char *name_buffer_glob = (state.actual_program.is_vertex()) ? VERTEX_UB_GROUP_NAME : FRAGMENT_UB_GROUP_NAME;

        std::uint32_t base = 0;
        for (std::size_t i = 0; i < input.uniform_buffers.size(); i++) {
            std::uint32_t real_size = static_cast<std::uint32_t>((input.uniform_buffers[i].size + 3) / 4 * 16);
            writer.add_declaration(fmt::format("if ((offset >= {}) && (offset < {})) {{", base, base + real_size));
            writer.indent_declaration();
            if (base != 0)
                writer.add_declaration(fmt::format("offset -= {};", base));
            writer.add_declaration("int vec_index = offset / 16;");
            writer.add_declaration("int bytes_in_vec = offset - vec_index * 16;");
            writer.add_declaration("int comp_in_vec = bytes_in_vec / 4;");
            writer.add_declaration("int start_byte_in_comp = bytes_in_vec - comp_in_vec * 4;");
            writer.add_declaration("int lshift_amount = 4 - start_byte_in_comp;");
            writer.add_declaration("int next_comp = comp_in_vec + 1;");
            writer.add_declaration("bool aligned = (lshift_amount == 4);");
            writer.add_declaration(fmt::format("return uintBitsToFloat((floatBitsToUint({}.buffer{}[vec_index][comp_in_vec]) << uint(start_byte_in_comp * 8))"
                                               " | (aligned ? 0u : (floatBitsToUint({}.buffer{}[vec_index + (next_comp / 4)][next_comp - (next_comp / 4) * 4]) >> uint((aligned ? 0 : lshift_amount) * 8))));",
                name_buffer_glob, input.uniform_buffers[i].index, name_buffer_glob, input.uniform_buffers[i].index));
            writer.dedent_declaration();
            writer.add_declaration("}");
            base += real_size;
        }

        writer.add_declaration("return 0.0;");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }
}
} // namespace shader::usse::glsl

namespace shader {
static std::string translate_color_format(const SceGxmColorBaseFormat format) {
    switch (format) {
    case SCE_GXM_COLOR_BASE_FORMAT_U8U8U8U8:
        return "rgba8";

    case SCE_GXM_COLOR_BASE_FORMAT_S8S8S8S8:
        return "rgba8_snorm";

    case SCE_GXM_COLOR_BASE_FORMAT_F16F16F16F16:
        return "rgba16f";

    case SCE_GXM_COLOR_BASE_FORMAT_U2U10U10U10:
        return "rgb10_a2";

    case SCE_GXM_COLOR_BASE_FORMAT_F11F11F10:
        return "r11f_g11f_b10f";

    case SCE_GXM_COLOR_BASE_FORMAT_F32F32:
        return "rg32f";

    default:
        return "rgba8";
    }
}

static std::string convert_gxp_to_glsl_impl(const SceGxmProgram &program, const std::string &shader_hash, const FeatureState &features, TranslationState &translation_state, const Hints *hints,
    bool force_shader_debug, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper) {
    usse::glsl::ProgramState program_state{ program };
    usse::glsl::CodeWriter code_writer;

    usse::ProgramInput inputs = shader::get_program_input(program);
    usse::glsl::ShaderVariables variables(code_writer, inputs, program.is_vertex());

    std::stringstream disasm_dump;
    usse::disasm::disasm_storage = &disasm_dump;

    usse::glsl::create_necessary_headers(code_writer, program, features, shader_hash);
    usse::glsl::create_parameters(program_state, code_writer, variables, features, translation_state, program, inputs, hints);

    usse::glsl::USSERecompilerGLSL recomp(program_state, code_writer, variables, features);

    if (program.secondary_program_instr_count) {
        recomp.visitor->set_secondary_program(true);
        recomp.reset(program.secondary_program_start(), program.secondary_program_instr_count);
        recomp.compile_to_function();
    }

    recomp.visitor->set_secondary_program(false);
    recomp.reset(program.primary_program_start(), program.primary_program_instr_count);
    recomp.compile_to_function();

    code_writer.set_active_body_type(usse::glsl::BODY_TYPE_POSTWORK);

    usse::glsl::create_output(program_state, code_writer, variables, features, program, hints);
    usse::glsl::create_necessary_footers(code_writer, program, features);

    code_writer.set_active_body_type(usse::glsl::BODY_TYPE_MAIN);

    usse::glsl::create_program_needed_functions(program_state, inputs, code_writer);

    variables.generate_helper_functions();
    std::string final_result = code_writer.assemble();

    if (dumper) {
        dumper(program.is_vertex() ? "vert" : "frag", final_result);
        dumper("dsm", disasm_dump.str());
    }

    LOG_INFO("GLSL Code:\n {}", final_result);

    return final_result;
}

GeneratedShader convert_gxp_to_glsl(const SceGxmProgram &program, const std::string &shader_hash, const FeatureState &features, const Hints &hints, bool maskupdate,
    bool force_shader_debug, const std::function<bool(const std::string &ext, const std::string &dump)> &dumper) {
    TranslationState translation_state;

    if (!features.support_unknown_format) {
        // take the color format of the current surface, hoping the shader is not used on two surfaces with different formats (this should be the case)
        translation_state.image_storage_format = translate_color_format(gxm::get_base_format(hints.color_format));
    }

    GeneratedShader shader{};
    shader.glsl = convert_gxp_to_glsl_impl(program, shader_hash, features, translation_state, &hints, force_shader_debug, dumper);
    return shader;
}
} // namespace shader