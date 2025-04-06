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

#include <shader/constant_table.h>
#include <shader/glsl/code_writer.h>
#include <shader/glsl/consts.h>
#include <shader/glsl/params.h>

#include <util/bit_cast.h>
#include <util/log.h>
#include <util/string_utils.h>

#include <fmt/format.h>

#include <queue>

namespace shader::usse::glsl {
ShaderVariables::ShaderVariables(CodeWriter &writer, const ProgramInput &input, const bool is_vertex)
    : writer(writer)
    , program_input(input)
    , pa_bank("pa")
    , sa_bank("sa")
    , internal_bank("i")
    , temp_bank("r")
    , output_bank("o")
    , is_vertex(is_vertex) {
    index_variables[0] = index_variables[1] = false;
    should_gen_vecfloat_bitcast[0] = should_gen_vecfloat_bitcast[1] = should_gen_vecfloat_bitcast[2]
        = should_gen_vecfloat_bitcast[3] = false;
    should_gen_clamp16 = false;
    should_gen_textureprojcube = false;
    should_gen_texture_get_bilinear_coefficients = false;
    should_gen_indexing_lookup[0] = should_gen_indexing_lookup[1] = should_gen_indexing_lookup[2]
        = should_gen_indexing_lookup[3] = false;

    std::memset(should_gen_pack_unpack, 0, sizeof(should_gen_pack_unpack));
}

ShaderVariableVec4Info::ShaderVariableVec4Info() {
    dirty[0] = dirty[1] = dirty[2] = dirty[3] = -1;
}

ShaderVariableBank *ShaderVariables::get_reg_bank(const RegisterBank bank) {
    switch (bank) {
    case RegisterBank::PRIMATTR:
        return &pa_bank;
    case RegisterBank::SECATTR:
        return &sa_bank;
    case RegisterBank::FPINTERNAL:
        return &internal_bank;
    case RegisterBank::TEMP:
        return &temp_bank;
    case RegisterBank::OUTPUT:
        return &output_bank;
    default:
        // LOG_WARN("Reg bank {} unsupported", static_cast<uint8_t>(reg_bank));
        return nullptr;
    }
}

static std::string generate_read_vector(const std::string &source, const int offset, const Swizzle4 &swizz, const std::uint8_t dest_mask, DataType dt, const bool is_gpi = false) {
    bool swizzle_all_var = true;
    std::string swizzle_str;

    for (std::uint8_t i = 0; i < 4; i++) {
        if (dest_mask & (1 << i)) {
            if (swizz[i] > SwizzleChannel::C_W) {
                swizzle_all_var = false;
                break;
            } else {
                swizzle_str += ('w' + (static_cast<char>(static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X)) + 1) % 4);
            }
        }
    }

    if (is_gpi || (offset % 4 == 0)) {
        if (swizzle_all_var) {
            if (dest_mask == 0b1111 && swizz == Swizzle4(SWIZZLE_CHANNEL_4_DEFAULT)) {
                return fmt::format("{}{}", source, offset);
            } else {
                return fmt::format("{}{}.{}", source, offset, swizzle_str);
            }
        }
    }

    std::string grouped;
    std::uint8_t comp_count = 0;
    std::string result = "";
    std::uint8_t already_scanned = 0;

    // 1. Unaligned, but we can work it out to make it look more nicely
    // 2. Just contains 0 and 1 somewhere
    int still_can_swizzle_out_count = 0;
    int mod = is_gpi ? 0 : (offset % 4);
    int offset_clean = is_gpi ? offset : (offset / 4 * 4);

    for (std::uint8_t i = 0; i < 4; i++) {
        if (dest_mask & (1 << i)) {
            if ((swizz[i] <= SwizzleChannel::C_W) && (static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X) + mod < 4)) {
                still_can_swizzle_out_count++;
            } else {
                break;
            }
        }
    }
    if ((still_can_swizzle_out_count == 0) && swizzle_all_var) {
        // Permanent resident to the next offset. Well no problem
        for (std::size_t i = 0; i < swizzle_str.length(); i++) {
            swizzle_str[i] = static_cast<char>('w' + ((swizzle_str[i] - 'w' + mod) % 4));
        }

        return fmt::format("{}{}.{}", source, offset_clean + 4, swizzle_str);
    } else {
        // If it's 1 then just follow the plan below
        if (still_can_swizzle_out_count > 1) {
            // Relocate
            result = swizzle_str.substr(0, still_can_swizzle_out_count);
            for (std::size_t i = 0; i < result.length(); i++) {
                result[i] = static_cast<char>('w' + (result[i] - 'w' + mod) % 4);
            }
            result = fmt::format("{}{}.{}, ", source, offset_clean, result);
            already_scanned = static_cast<std::uint8_t>(still_can_swizzle_out_count);
        }
    }

    std::uint8_t in_scan_count = 0;
    bool newly_scanned = false;
    std::string postfix;

    if (is_unsigned_integer_data_type(dt)) {
        postfix = "u";
    } else if (is_float_data_type(dt)) {
        postfix = ".0";
    }

    for (std::uint8_t i = 0; i < 4; i++) {
        if (dest_mask & (1 << i)) {
            in_scan_count++;

            if (in_scan_count > already_scanned) {
                newly_scanned = true;
                if (swizz[i] <= SwizzleChannel::C_W) {
                    int current_channel = (static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X) + mod);
                    result += fmt::format("{}{}.{}, ", source, (current_channel >= 4) ? offset_clean + 4 : offset_clean,
                        static_cast<char>('w' + (current_channel + 1) % 4));
                } else {
                    switch (swizz[i]) {
                    case SwizzleChannel::C_0:
                        result += std::string("0") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_1:
                        result += std::string("1") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_2:
                        result += std::string("2") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_H:
                        result += "0.5, ";
                        break;

                    default:
                        assert(false && "Unreachable");
                        LOG_ERROR("Invalid swizzle channel encountered!");

                        return "ERROR";
                    }
                }
            }
        }
    }

    // Remove last ", "
    result.pop_back();
    result.pop_back();

    if (in_scan_count == 1) {
        return result;
    }

    return newly_scanned ? fmt::format("vec{}({})", in_scan_count, result) : result;
}

void ShaderVariables::mark_f32_raw_dirty(const RegisterBank bank, const int offset) {
    get_reg_bank(bank)->variables[offset / 4 * 4].dirty[offset % 4] = 2;
}

void ShaderVariables::flush_bank(ShaderVariableBank &bank) {
    std::queue<RestorationInfo> restorations;

    for (auto &[base, var] : bank.variables) {
        if ((var.declared & ShaderVariableVec4Info::DECLARED_FLOAT4) == 0) {
            writer.add_declaration(fmt::format("vec4 {}{};", bank.prefix, base));
            var.declared |= ShaderVariableVec4Info::DECLARED_FLOAT4;
        }

        for (int i = 0; i < 4; i++) {
            // Force to load F32 after flush. One of the reason is because it might got declared inside an inner
            // block or not, and then the block after non conditionally for example, just continue using it as normal.
            // it's uncertain
            if (var.dirty[i] != -1) {
                // Flush and make it as new!
                restorations.push({ var, i, var.dirty[i], base, var.declared });
            }
            var.dirty[i] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
        }
    }

    if (!restorations.empty())
        do_restore(restorations, bank.prefix);
}

void ShaderVariables::flush() {
    // TODO: I think manual loop is very inefficient. But what else. Haha
    flush_bank(temp_bank);
    flush_bank(sa_bank);
    flush_bank(pa_bank);
    flush_bank(output_bank);
}

void ShaderVariables::do_restore(std::queue<RestorationInfo> &restore_infos, const std::string &prefix) {
    // Process the restoration queue
    while (!restore_infos.empty()) {
        RestorationInfo &res_info = restore_infos.front();
        int var_key_base = res_info.base_current;
        int odd_index = res_info.component;
        if (odd_index >= 4) {
            var_key_base += 4;
            odd_index %= 4;
        }

        switch (res_info.original_type) {
        // Separate them because of casting math (some drives just auto cast negative number to 0, dont even care about bit pattern)
        // If we all assume it to be int then it's gonna be a problem to sync them back later.
        case ShaderVariableVec4Info::DIRTY_TYPE_S8:
        case ShaderVariableVec4Info::DIRTY_TYPE_U8: {
            char prefix_for_type = (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) ? 'i' : 'u';
            char prefix_for_pack = (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) ? 'S' : 'U';

            std::uint32_t index_for_u8 = (res_info.base_current + res_info.component) * 4;

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) {
                    should_gen_pack_unpack[GEN_PACK_4XS8] = true;
                } else {
                    should_gen_pack_unpack[GEN_PACK_4XU8] = true;
                }

                writer.add_to_current_body(fmt::format("{}{}.{} = pack4x{}8({}8{}{});", prefix, var_key_base,
                    static_cast<char>('w' + (odd_index + 1) % 4), prefix_for_pack, prefix_for_type,
                    prefix, index_for_u8));
            }

            std::uint32_t mask_other_8bit = ((res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_U8) ? (1 << (ShaderVariableVec4Info::DECLARED_S8_SHIFT_START + odd_index))
                                                                                                               : (1 << (ShaderVariableVec4Info::DECLARED_U8_SHIFT_START + odd_index)));

            if (res_info.decl & mask_other_8bit) {
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) {
                    writer.add_to_current_body(fmt::format("u8{}{} = uvec4(i8{}{});", prefix, index_for_u8, prefix, index_for_u8));
                } else {
                    writer.add_to_current_body(fmt::format("i8{}{} = ivec4(u8{}{});", prefix, index_for_u8, prefix, index_for_u8));
                }
            }

            int index_16 = (res_info.base_current + res_info.component) * 2;
            int unaligned = index_16 % 4;
            index_16 = (index_16 / 4) * 4;

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_HALF4_N2 : ShaderVariableVec4Info::DECLARED_HALF4_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XF16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("h16{}{}.{} = unpack2xF16({}{}.{});", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) {
                        should_gen_pack_unpack[GEN_PACK_4XS8] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_4XU8] = true;
                    }

                    writer.add_to_current_body(fmt::format("h16{}{}.{} = unpack2xF16(pack4x{}8({}8{}{}));", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix_for_pack, prefix_for_type, prefix, (res_info.base_current + res_info.component) * 4));
                }
            }

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_U16_N2 : ShaderVariableVec4Info::DECLARED_U16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XU16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("u16{}{}.{} = unpack2xU16({}{}.{});", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) {
                        should_gen_pack_unpack[GEN_PACK_4XS8] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_4XU8] = true;
                    }

                    writer.add_to_current_body(fmt::format("u16{}{}.{} = unpack2xU16(pack4x{}8({}8{}{}));", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix_for_pack, prefix_for_type, prefix, (res_info.base_current + res_info.component) * 4));
                }
            }

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_S16_N2 : ShaderVariableVec4Info::DECLARED_S16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XS16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("i16{}{}.{} = unpack2xS16({}{}.{});", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S8) {
                        should_gen_pack_unpack[GEN_PACK_4XS8] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_4XU8] = true;
                    }

                    writer.add_to_current_body(fmt::format("i16{}{}.{} = unpack2xS16(pack4x{}8({}8{}{}));", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix_for_pack, prefix_for_type, prefix, (res_info.base_current + res_info.component) * 4));
                }
            }

            break;
        }

        case ShaderVariableVec4Info::DIRTY_TYPE_F16: {
            int index_16 = (res_info.base_current + res_info.component) * 2;
            int index_8 = (res_info.base_current + res_info.component) * 4;
            int unaligned = index_16 % 4;
            bool use_part2 = (index_16 % 8) >= 4;
            index_16 = (index_16 / 4) * 4;

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                writer.add_to_current_body(fmt::format("{}{}.{} = pack2xF16(h16{}{}.{});", prefix, var_key_base,
                    static_cast<char>('w' + (odd_index + 1) % 4), prefix, index_16, unaligned ? "zw" : "xy"));
            }

            int shift_check_decl_s8 = ShaderVariableVec4Info::DECLARED_S8_SHIFT_START + odd_index;
            int shift_check_decl_u8 = ShaderVariableVec4Info::DECLARED_U8_SHIFT_START + odd_index;
            int shift_check_decl_fx10 = ShaderVariableVec4Info::DECLARED_FX10_SHIFT_START + odd_index;

            if (res_info.decl & (1 << shift_check_decl_s8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XS8] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("i8{}{} = unpack4xS8({}{}.{});", prefix, index_8,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                    writer.add_to_current_body(fmt::format("i8{}{} = unpack4xS8(pack2xF16(h16{}{}.{}));", prefix, index_8,
                        prefix, index_16, unaligned ? "zw" : "xy"));
                }
            }

            if (res_info.decl & (1 << shift_check_decl_u8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XU8] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("u8{}{} = unpack4xU8({}{}.{});", prefix, index_8,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                    writer.add_to_current_body(fmt::format("u8{}{} = unpack4xU8(pack2xF16(h16{}{}.{}));", prefix, index_8,
                        prefix, index_16, unaligned ? "zw" : "xy"));
                }
            }

            if (res_info.decl & (1 << shift_check_decl_fx10)) {
                if (shift_check_decl_fx10 != ShaderVariableVec4Info::DECLARED_FX10_N4) {
                    should_gen_pack_unpack[GEN_UNPACK_3XFX10] = true;

                    if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                        writer.add_to_current_body(fmt::format("fx10{}{}.xyz = unpack3xFX10({}{}.{});", prefix, index_8,
                            prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                    } else {
                        should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                        writer.add_to_current_body(fmt::format("fx10{}{}.xyz = unpack3xFX10(pack2xF16(h16{}{}.{}));", prefix, index_8,
                            prefix, index_16, unaligned ? "zw" : "xy"));
                    }
                } else {
                    LOG_ERROR("C10 data type can't have 4th element!");
                }
            }

            if (res_info.decl & (use_part2 ? ShaderVariableVec4Info::DECLARED_S16_N2 : ShaderVariableVec4Info::DECLARED_S16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XS16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("i16{}{}.{} = unpack2xS16({}{}.{});", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                    writer.add_to_current_body(fmt::format("i16{}{}.{} = unpack2xS16(pack2xF16(h16{}{}.{}));", prefix, index_8,
                        unaligned ? "zw" : "xy", prefix, index_16, unaligned ? "zw" : "xy"));
                }
            }

            if (res_info.decl & (use_part2 ? ShaderVariableVec4Info::DECLARED_U16_N2 : ShaderVariableVec4Info::DECLARED_U16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XU16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("u16{}{}.{} = unpack2xU16({}{}.{});", prefix, index_16,
                        unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    should_gen_pack_unpack[GEN_PACK_2XF16] = true;
                    writer.add_to_current_body(fmt::format("u16{}{}.{} = unpack2xU16(pack2xF16(h16{}{}.{}));", prefix, index_8,
                        unaligned ? "zw" : "xy", prefix, index_16, unaligned ? "zw" : "xy"));
                }
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_U16) {
                should_gen_pack_unpack[GEN_PACK_4XF16] = true;
                should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                writer.add_to_current_body(fmt::format("u16{}{} = unpack4xU16(pack4xF16(h16{}{}));", prefix, index_16, prefix, index_16));
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_S16) {
                should_gen_pack_unpack[GEN_PACK_4XF16] = true;
                should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                writer.add_to_current_body(fmt::format("i16{}{} = unpack4xS16(pack4xF16(h16{}{}));", prefix, index_16, prefix, index_16));
            }

            break;
        }

        case ShaderVariableVec4Info::DIRTY_TYPE_F32: {
            int index_16 = (res_info.base_current + res_info.component) * 2;
            int index_8 = (res_info.base_current + res_info.component) * 4;
            int unaligned = index_16 % 4;
            index_16 = (index_16 / 4) * 4;

            int shift_check_decl_s8 = ShaderVariableVec4Info::DECLARED_S8_SHIFT_START + odd_index;
            int shift_check_decl_u8 = ShaderVariableVec4Info::DECLARED_U8_SHIFT_START + odd_index;
            int shift_check_decl_fx10 = ShaderVariableVec4Info::DECLARED_FX10_SHIFT_START + odd_index;

            if (res_info.decl & (1 << shift_check_decl_s8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XS8] = true;
                writer.add_to_current_body(fmt::format("i8{}{} = unpack4xS8({}{}.{});", prefix, index_8,
                    prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
            }

            if (res_info.decl & (1 << shift_check_decl_u8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XU8] = true;
                writer.add_to_current_body(fmt::format("u8{}{} = unpack4xU8({}{}.{});", prefix, index_8,
                    prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
            }

            if (res_info.decl & (1 << shift_check_decl_fx10)) {
                if (shift_check_decl_fx10 != ShaderVariableVec4Info::DECLARED_FX10_N4) {
                    should_gen_pack_unpack[GEN_UNPACK_3XFX10] = true;
                    writer.add_to_current_body(fmt::format("fx10{}{}.xyz = unpack3xFX10({}{}.{});", prefix, index_8,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    LOG_ERROR("C10 data type can't have 4th element!");
                }
            }

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_HALF4_N2 : ShaderVariableVec4Info::DECLARED_HALF4_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XF16] = true;
                writer.add_to_current_body(fmt::format("h16{}{}.{} = unpack2xF16({}{}.{});", prefix, index_16,
                    unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
            }

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_U16_N2 : ShaderVariableVec4Info::DECLARED_U16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XU16] = true;
                writer.add_to_current_body(fmt::format("u16{}{}.{} = unpack2xU16({}{}.{});", prefix, index_16,
                    unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
            }

            if (res_info.decl & (((index_16 % 8) >= 4) ? ShaderVariableVec4Info::DECLARED_S16_N2 : ShaderVariableVec4Info::DECLARED_S16_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XS16] = true;
                writer.add_to_current_body(fmt::format("i16{}{}.{} = unpack2xS16({}{}.{});", prefix, index_16,
                    unaligned ? "zw" : "xy", prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_HALF4) {
                should_gen_pack_unpack[GEN_UNPACK_4XF16] = true;
                writer.add_to_current_body(fmt::format("h16{}{} = unpack4xF16({}{}.{}{});", prefix, index_16,
                    prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4), static_cast<char>('w' + (odd_index + 2) % 4)));
            }
            
            if (res_info.decl & ShaderVariableVec4Info::DECLARED_U16) {
                should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                writer.add_to_current_body(fmt::format("h16{}{} = unpack4xU16({}{}.{}{});", prefix, index_16,
                    prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4), static_cast<char>('w' + (odd_index + 2) % 4)));
            }
            
            if (res_info.decl & ShaderVariableVec4Info::DECLARED_S16) {
                should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                writer.add_to_current_body(fmt::format("h16{}{} = unpack4xS16({}{}.{}{});", prefix, index_16,
                    prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4), static_cast<char>('w' + (odd_index + 2) % 4)));
            }

            break;
        }

        case ShaderVariableVec4Info::DIRTY_TYPE_U16:
        case ShaderVariableVec4Info::DIRTY_TYPE_S16: {
            int index_16 = (res_info.base_current + res_info.component) * 2;
            int index_8 = (res_info.base_current + res_info.component) * 4;
            int unaligned = index_16 % 4;
            bool part2 = (index_16 % 8) >= 4;
            index_16 = (index_16 / 4) * 4;
            const char *swizzle_f16 = (unaligned ? "zw" : "xy");

            char prefix_for_type = (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) ? 'i' : 'u';
            char prefix_for_pack = (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) ? 'S' : 'U';

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                    should_gen_pack_unpack[GEN_PACK_2XS16] = true;
                } else {
                    should_gen_pack_unpack[GEN_PACK_2XU16] = true;
                }
                writer.add_to_current_body(fmt::format("{}{}.{} = pack2x{}16({}16{}{}.{});", prefix, var_key_base,
                    static_cast<char>('w' + (odd_index + 1) % 4), prefix_for_pack, prefix_for_type, prefix, index_16, swizzle_f16));
            }

            std::uint32_t mask_check_other = 0;
            if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                mask_check_other = (part2 ? ShaderVariableVec4Info::DECLARED_U16_N2 : ShaderVariableVec4Info::DECLARED_U16_N1);
            } else {
                mask_check_other = (part2 ? ShaderVariableVec4Info::DECLARED_S16_N2 : ShaderVariableVec4Info::DECLARED_S16_N1);
            }

            if (res_info.decl & mask_check_other) {
                // Dirty hack to convert between signed and unsigned 16-bit values since uvec2 and ivec2 are 32-bit
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                    writer.add_to_current_body(fmt::format("u16{}{}.{} = uvec2(i16{}{}.{} << 16) >> 16;", prefix, index_16, swizzle_f16, prefix, index_16, swizzle_f16));
                } else {
                    writer.add_to_current_body(fmt::format("i16{}{}.{} = ivec2(u16{}{}.{} << 16) >> 16;",  prefix, index_16, swizzle_f16, prefix, index_16, swizzle_f16));
                }
            }

            if (res_info.decl & (part2 ? ShaderVariableVec4Info::DECLARED_HALF4_N2 : ShaderVariableVec4Info::DECLARED_HALF4_N1)) {
                should_gen_pack_unpack[GEN_UNPACK_2XF16] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("h16{}{}.{} = unpack2xF16({}{}.{});", prefix, index_16, swizzle_f16,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                        should_gen_pack_unpack[GEN_PACK_2XS16] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_2XU16] = true;
                    }

                    writer.add_to_current_body(fmt::format("h16{}{}.{} = unpack2xF16(pack2x{}16({}16{}{}.{}));", prefix, index_16,
                        swizzle_f16, prefix_for_pack, prefix_for_type, prefix, index_16, swizzle_f16));
                }
            }

            int shift_check_decl_s8 = ShaderVariableVec4Info::DECLARED_S8_SHIFT_START + odd_index;
            int shift_check_decl_u8 = ShaderVariableVec4Info::DECLARED_U8_SHIFT_START + odd_index;
            int shift_check_decl_fx10 = ShaderVariableVec4Info::DECLARED_FX10_SHIFT_START + odd_index;

            if (res_info.decl & (1 << shift_check_decl_s8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XS8] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("i8{}{} = unpack4xS8({}{}.{}));", prefix, index_8,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                        should_gen_pack_unpack[GEN_PACK_2XS16] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_2XU16] = true;
                    }
                    writer.add_to_current_body(fmt::format("i8{}{} = unpack4xS8(pack2x{}16({}16{}{}.{}));", prefix, index_8,
                        prefix_for_pack, prefix_for_type, prefix, index_16, swizzle_f16));
                }
            }

            if (res_info.decl & (1 << shift_check_decl_u8)) {
                should_gen_pack_unpack[GEN_UNPACK_4XU8] = true;

                if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    writer.add_to_current_body(fmt::format("u8{}{} = unpack4xU8({}{}.{}));", prefix, index_8,
                        prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                } else {
                    if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                        should_gen_pack_unpack[GEN_PACK_2XS16] = true;
                    } else {
                        should_gen_pack_unpack[GEN_PACK_2XU16] = true;
                    }
                    writer.add_to_current_body(fmt::format("u8{}{} = unpack4xU8(pack2x{}16({}16{}{}.{}));", prefix, index_8,
                        prefix_for_pack, prefix_for_type, prefix, index_16, swizzle_f16));
                }
            }

            if (res_info.decl & (1 << shift_check_decl_fx10)) {
                if (shift_check_decl_fx10 != ShaderVariableVec4Info::DECLARED_FX10_N4) {
                    should_gen_pack_unpack[GEN_UNPACK_3XFX10] = true;

                    if (res_info.decl & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                        writer.add_to_current_body(fmt::format("fx10{}{}.xyz = unpack3xFX10({}{}.{}));", prefix, index_8,
                            prefix, var_key_base, static_cast<char>('w' + (odd_index + 1) % 4)));
                    } else {
                        if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                            should_gen_pack_unpack[GEN_PACK_2XS16] = true;
                        } else {
                            should_gen_pack_unpack[GEN_PACK_2XU16] = true;
                        }
                        writer.add_to_current_body(fmt::format("fx10{}{}.xyz = unpack3xFX10(pack2x{}16({}16{}{}.{}));", prefix, index_8,
                            prefix_for_pack, prefix_for_type, prefix, index_16, swizzle_f16));
                    }
                } else {
                    LOG_ERROR("C10 data type can't have 4th element!");
                }
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_HALF4) {
                should_gen_pack_unpack[GEN_UNPACK_4XF16] = true;

                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                    should_gen_pack_unpack[GEN_PACK_4XS16] = true;
                    writer.add_to_current_body(fmt::format("h16{}{} = unpack4xF16(pack4xS16(i16{}{}));", prefix, index_16, prefix, index_16));
                } else {
                    should_gen_pack_unpack[GEN_PACK_4XU16] = true;
                    writer.add_to_current_body(fmt::format("h16{}{} = unpack4xF16(pack4xU16(u16{}{}));", prefix, index_16, prefix, index_16));
                }
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_U16) {
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_S16) {
                    should_gen_pack_unpack[GEN_PACK_4XS16] = true;
                    should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                    writer.add_to_current_body(fmt::format("h16{}{} = unpack4xU16(pack4xS16(i16{}{}));", prefix, index_16, prefix, index_16));
                }
            }

            if (res_info.decl & ShaderVariableVec4Info::DECLARED_S16) {
                if (res_info.original_type == ShaderVariableVec4Info::DIRTY_TYPE_U16) {
                    should_gen_pack_unpack[GEN_PACK_4XU16] = true;
                    should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                    writer.add_to_current_body(fmt::format("h16{}{} = unpack4xS16(pack4xU16(u16{}{}));", prefix, index_16, prefix, index_16));
                }
            }

            break;
        }

        default:
            break;
        }

        restore_infos.pop();
    }
}

void ShaderVariables::prepare_variables(ShaderVariableBank &target_bank, const DataType dt, const RegisterBank bank_type, const int index, const int shift, const Swizzle4 &swizz,
    const std::uint8_t dest_mask, const bool for_write, std::string &output_prefix, int &real_load_index, bool &load_to_another_vec) {
    int shifted_load_index = index + shift;
    real_load_index = shifted_load_index;

    int var_key = shifted_load_index / 4 * 4;

    if (bank_type == RegisterBank::FPINTERNAL) {
        var_key = real_load_index;

        // Never need load anew
        if (!target_bank.variables[var_key].declared) {
            writer.add_declaration(fmt::format("vec4 i{};", var_key));
            target_bank.variables[var_key].declared = 1;
        }

        output_prefix = target_bank.prefix;
    } else {
        load_to_another_vec = false;

        auto calc_load_to_another_vec = [&]() {
            if (real_load_index % 4 != 0) {
                for (std::uint8_t i = 0; i < 4; i++) {
                    if (dest_mask & (1 << i)) {
                        if (swizz[i] <= SwizzleChannel::C_W && ((static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X) + (real_load_index % 4)) >= 4)) {
                            load_to_another_vec = true;
                            break;
                        }
                    }
                }
            }
        };

        switch (get_data_type_size(dt)) {
        case 1: {
            if (dt != DataType::C10) {
                output_prefix = ((dt == DataType::UINT8) ? std::string("u8") : std::string("i8")) + target_bank.prefix;

                std::uint32_t shift_amount = ((dt == DataType::UINT8) ? (ShaderVariableVec4Info::DECLARED_U8_SHIFT_START + (shifted_load_index % 4))
                                                                      : (ShaderVariableVec4Info::DECLARED_S8_SHIFT_START + (shifted_load_index % 4)));
                std::uint32_t shift_mask = 1 << shift_amount;
                real_load_index *= 4;
                if ((target_bank.variables[var_key].declared & shift_mask) == 0) {
                    writer.add_declaration(fmt::format("{}vec4 {}{};", output_prefix[0], output_prefix, real_load_index));
                    if (target_bank.variables[var_key].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                        writer.add_to_current_body(fmt::format("{}{} = unpack4x{}8({}{}.{});", output_prefix, real_load_index, (dt == DataType::UINT8) ? 'U' : 'S',
                            target_bank.prefix, var_key, static_cast<char>('w' + (shifted_load_index + 1) % 4)));

                        if (dt == DataType::UINT8) {
                            should_gen_pack_unpack[GEN_UNPACK_4XU8] = true;
                        } else {
                            should_gen_pack_unpack[GEN_UNPACK_4XS8] = true;
                        }
                    }
                    target_bank.variables[var_key].declared |= shift_mask;
                }
            } else {
                output_prefix = std::string("fx10") + target_bank.prefix;

                std::uint32_t shift_amount = ShaderVariableVec4Info::DECLARED_FX10_SHIFT_START + (shifted_load_index % 4);
                std::uint32_t shift_mask = 1 << shift_amount;
                real_load_index *= 4;
                if ((target_bank.variables[var_key].declared & shift_mask) == 0) {
                    if (shift_mask != ShaderVariableVec4Info::DECLARED_FX10_N4) {
                        writer.add_declaration(fmt::format("ivec4 {}{};", output_prefix, real_load_index));
                        if (target_bank.variables[var_key].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                            writer.add_to_current_body(fmt::format("{}{}.xyz = unpack3xFX10({}{}.{});", output_prefix, real_load_index,
                                target_bank.prefix, var_key, static_cast<char>('w' + (shifted_load_index + 1) % 4)));

                            should_gen_pack_unpack[GEN_UNPACK_3XFX10] = true;
                        }
                        target_bank.variables[var_key].declared |= shift_mask;
                    } else {
                        LOG_ERROR("C10 data type can't have 4th element!");
                    }
                }
            }
            break;
        }

        case 2: {
            std::uint32_t mask_decl_2 = 0;
            std::uint32_t mask_decl_1 = 0;
            std::string vprefix = "";
            switch (dt) {
            case DataType::F16:
                output_prefix = std::string("h16") + target_bank.prefix;
                mask_decl_1 = ShaderVariableVec4Info::DECLARED_HALF4_N1;
                mask_decl_2 = ShaderVariableVec4Info::DECLARED_HALF4_N2;
                break;

            case DataType::UINT16:
                output_prefix = std::string("u16") + target_bank.prefix;
                mask_decl_1 = ShaderVariableVec4Info::DECLARED_U16_N1;
                mask_decl_2 = ShaderVariableVec4Info::DECLARED_U16_N2;
                vprefix = "u";
                break;

            case DataType::INT16:
                output_prefix = std::string("i16") + target_bank.prefix;
                mask_decl_1 = ShaderVariableVec4Info::DECLARED_S16_N1;
                mask_decl_2 = ShaderVariableVec4Info::DECLARED_S16_N2;
                vprefix = "i";
                break;

            default:
                break;
            }
            real_load_index *= 2;
            calc_load_to_another_vec();
            // Try to load the latter
            bool load_begin = (real_load_index % 8) < 4;
            if (load_begin && ((target_bank.variables[var_key].declared & mask_decl_1) == 0)) {
                writer.add_declaration(fmt::format("{}vec4 {}{};", vprefix, output_prefix, (real_load_index / 4) * 4));
                target_bank.variables[var_key].declared |= mask_decl_1;

                if (target_bank.variables[var_key].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    if (dt == DataType::F16) {
                        should_gen_pack_unpack[GEN_UNPACK_4XF16] = true;
                        writer.add_to_current_body(fmt::format("{}{} = unpack4xF16({}{}.xy);", output_prefix,
                            real_load_index / 4 * 4, target_bank.prefix, var_key));
                    } else {
                        if (dt == DataType::UINT16) {
                            should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                        } else {
                            should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                        }
                        writer.add_to_current_body(fmt::format("{}{} = unpack4x{}16({}{}.xy);", output_prefix,
                            real_load_index / 4 * 4, (dt == DataType::UINT16) ? 'U' : 'S', target_bank.prefix, var_key));
                    }
                }
            }
            if ((!load_begin || load_to_another_vec) && ((target_bank.variables[var_key].declared & mask_decl_2) == 0)) {
                writer.add_declaration(fmt::format("{}vec4 {}{};", vprefix, output_prefix, (real_load_index / 4 + (load_begin && load_to_another_vec ? 1 : 0)) * 4));
                target_bank.variables[var_key].declared |= mask_decl_2;

                if (target_bank.variables[var_key].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    if (dt == DataType::F16) {
                        should_gen_pack_unpack[GEN_UNPACK_4XF16] = true;
                        writer.add_to_current_body(fmt::format("{}{} = unpack4xF16({}{}.zw);", output_prefix,
                            (real_load_index / 4 + (load_begin && load_to_another_vec ? 1 : 0)) * 4, target_bank.prefix, var_key));
                    } else {
                        if (dt == DataType::UINT16) {
                            should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                        } else {
                            should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                        }
                        writer.add_to_current_body(fmt::format("{}{} = unpack4x{}16({}{}.zw);", output_prefix,
                            (real_load_index / 4 + (load_begin && load_to_another_vec ? 1 : 0)) * 4, (dt == DataType::UINT16) ? 'U' : 'S', target_bank.prefix, var_key));
                    }
                }
            }
            if (load_to_another_vec && !load_begin && ((target_bank.variables[var_key + 4].declared & mask_decl_1) == 0)) {
                writer.add_declaration(fmt::format("{}vec4 {}{};", vprefix, output_prefix, (real_load_index / 4 + 1) * 4));
                target_bank.variables[var_key + 4].declared |= mask_decl_1;

                if (target_bank.variables[var_key + 4].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) {
                    if (dt == DataType::F16) {
                        should_gen_pack_unpack[GEN_UNPACK_4XF16] = true;
                        writer.add_to_current_body(fmt::format("{}{} = unpack4xF16({}{}.xy);", output_prefix,
                            real_load_index / 4 * 4 + 4, target_bank.prefix, var_key + 4));
                    } else {
                        if (dt == DataType::UINT16) {
                            should_gen_pack_unpack[GEN_UNPACK_4XU16] = true;
                        } else {
                            should_gen_pack_unpack[GEN_UNPACK_4XS16] = true;
                        }
                        writer.add_to_current_body(fmt::format("{}{} = unpack4x{}16({}{}.xy);", output_prefix,
                            real_load_index / 4 * 4 + 4, (dt == DataType::UINT16) ? 'U' : 'S', target_bank.prefix, var_key + 4));
                    }
                }
            }
            break;
        }

        case 4:
            output_prefix = target_bank.prefix;
            calc_load_to_another_vec();
            if ((target_bank.variables[var_key].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) == 0) {
                writer.add_declaration(fmt::format("vec4 {}{};", output_prefix, (shifted_load_index / 4) * 4));
                target_bank.variables[var_key].declared |= 1;
            }
            if (load_to_another_vec && ((target_bank.variables[var_key + 4].declared & ShaderVariableVec4Info::DECLARED_FLOAT4) == 0)) {
                writer.add_declaration(fmt::format("vec4 {}{};", output_prefix, (shifted_load_index / 4 + 1) * 4));
                target_bank.variables[var_key + 4].declared |= 1;
            }
            break;

        default:
            assert(false && "Unreachable");
            LOG_ERROR("Register access data size exceed 4!");
        }

        if (bank_type == RegisterBank::SECATTR) {
            auto copy_ub_data_to_target_sa = [&](const int sa_num) {
                int rounded_index = sa_num / 4 * 4;
                if (!target_bank.variables[rounded_index].inited) {
                    target_bank.variables[rounded_index].inited = true;

                    // We only sometimes copy data out when it's needed. PA is always copied
                    for (auto &buffer : program_input.uniform_buffers) {
                        if ((sa_num >= buffer.reg_start_offset) && (sa_num < (buffer.reg_start_offset + buffer.reg_block_size))) {
                            std::string buffer_access = fmt::format("{}.{}", is_vertex ? VERTEX_UB_GROUP_NAME : FRAGMENT_UB_GROUP_NAME,
                                fmt::format("{}{}", UB_MEMBER_NAME_FORMAT, buffer.index));

                            // Copy it out somewhere somewhere
                            if (buffer.reg_start_offset % 4 == 0) {
                                for (std::uint8_t i = 0; i < 4; i++) {
                                    target_bank.variables[rounded_index].dirty[i] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                }
                                writer.add_to_preload(fmt::format("sa{} = {}[{}];", rounded_index, buffer_access, (sa_num - buffer.reg_start_offset) / 4));
                            } else {
                                switch (buffer.reg_start_offset % 4) {
                                case 1:
                                    target_bank.variables[rounded_index].dirty[1] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                    target_bank.variables[rounded_index].dirty[2] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                    target_bank.variables[rounded_index].dirty[3] = ShaderVariableVec4Info::DIRTY_TYPE_F32;

                                    writer.add_to_preload(fmt::format("sa{}.yzw = {}[{}].xyz;", rounded_index, buffer_access, (sa_num - buffer.reg_start_offset) / 4,
                                        rounded_index + 4, buffer_access, (sa_num + 3 - buffer.reg_start_offset) / 4));

                                    if ((rounded_index + 4) < (buffer.reg_start_offset + buffer.reg_block_size)) {
                                        target_bank.variables[rounded_index + 4].dirty[0] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                        writer.add_to_preload(fmt::format("sa{}.x = {}[{}].w;", rounded_index + 4, buffer_access, (sa_num + 3 - buffer.reg_start_offset) / 4));
                                    }

                                    break;

                                case 2:
                                    target_bank.variables[rounded_index].dirty[2] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                    target_bank.variables[rounded_index].dirty[3] = ShaderVariableVec4Info::DIRTY_TYPE_F32;

                                    writer.add_to_preload(fmt::format("sa{}.zw = {}[{}].xy;", rounded_index, buffer_access, (sa_num - buffer.reg_start_offset) / 4));

                                    if ((rounded_index + 4) < (buffer.reg_start_offset + buffer.reg_block_size)) {
                                        target_bank.variables[rounded_index + 4].dirty[0] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                        target_bank.variables[rounded_index + 4].dirty[1] = ShaderVariableVec4Info::DIRTY_TYPE_F32;

                                        writer.add_to_preload(fmt::format("sa{}.xy = {}[{}].zw;", rounded_index + 4, buffer_access, (sa_num + 3 - buffer.reg_start_offset) / 4));
                                    }

                                    break;

                                case 3:
                                    target_bank.variables[rounded_index].dirty[3] = ShaderVariableVec4Info::DIRTY_TYPE_F32;

                                    writer.add_to_preload(fmt::format("sa{}.w = {}[{}].x;", rounded_index, buffer_access, (shifted_load_index - buffer.reg_start_offset) / 4));

                                    if ((rounded_index + 4) < (buffer.reg_start_offset + buffer.reg_block_size)) {
                                        target_bank.variables[rounded_index + 4].dirty[0] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                        target_bank.variables[rounded_index + 4].dirty[1] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                        target_bank.variables[rounded_index + 4].dirty[2] = ShaderVariableVec4Info::DIRTY_TYPE_F32;

                                        writer.add_to_preload(fmt::format("sa{}.xyz = {}[{}].yzw;", rounded_index + 4, buffer_access, (shifted_load_index + 3 - buffer.reg_start_offset) / 4));
                                    }
                                    break;

                                default:
                                    assert(false && "Unreachable");
                                    break;
                                }
                            }
                        }
                    }
                }
            };

            copy_ub_data_to_target_sa(shifted_load_index);
            if (load_to_another_vec) {
                copy_ub_data_to_target_sa(shifted_load_index + 4);
            }
        }

        ShaderVariableVec4Info &info = target_bank.variables[var_key];
        ShaderVariableVec4Info &info_next = target_bank.variables[var_key + 4];
        std::uint8_t data_type_size = get_data_type_size(dt);
        int write_type = 0;

        std::queue<RestorationInfo> restore_infos;
        int odd_index = shifted_load_index % 4;

        int type_dirty = 0;
        switch (dt) {
        case DataType::UINT8:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_U8;
            break;

        case DataType::INT8:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_S8;
            break;

        case DataType::UINT16:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_U16;
            break;

        case DataType::INT16:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_S16;
            break;

        case DataType::F16:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_F16;
            break;

        case DataType::F32:
            type_dirty = ShaderVariableVec4Info::DIRTY_TYPE_F32;
            break;

        default:
            break;
        }

        switch (data_type_size) {
        case 1:
            if ((info.dirty[odd_index] != type_dirty) && (info.dirty[odd_index] != -1)) {
                restore_infos.push({ info, odd_index, info.dirty[odd_index], var_key, info.declared });
                info.dirty[odd_index] = -1;
            }
            break;

        case 2: {
            bool has_low = false;
            bool has_high = false;

            for (std::uint8_t i = 0; i < 4; i++) {
                if (dest_mask & (1 << i)) {
                    if (swizz[i] <= SwizzleChannel::C_Y) {
                        has_low = true;
                    } else if (swizz[i] <= SwizzleChannel::C_W) {
                        has_high = true;
                    }
                }
            }

            if (has_low && has_high && (odd_index % 2) == 0 && info.dirty[odd_index] == info.dirty[odd_index + 1] && ((info.dirty[odd_index] != type_dirty) && (info.dirty[odd_index] != -1))) {
                int and_decl = 0;
                int or_decl = 0;
                if (info.declared & (ShaderVariableVec4Info::DECLARED_HALF4_N2 | ShaderVariableVec4Info::DECLARED_HALF4_N1)) {
                    and_decl |= ShaderVariableVec4Info::DECLARED_HALF4_N2 | ShaderVariableVec4Info::DECLARED_HALF4_N1;
                    or_decl |= ShaderVariableVec4Info::DECLARED_HALF4;
                }

                if (info.declared & (ShaderVariableVec4Info::DECLARED_U16_N2 | ShaderVariableVec4Info::DECLARED_U16_N1)) {
                    and_decl |= ShaderVariableVec4Info::DECLARED_U16_N2 | ShaderVariableVec4Info::DECLARED_U16_N1;
                    or_decl |= ShaderVariableVec4Info::DECLARED_U16;
                }

                if (info.declared & (ShaderVariableVec4Info::DECLARED_S16_N2 | ShaderVariableVec4Info::DECLARED_S16_N1)) {
                    and_decl |= ShaderVariableVec4Info::DECLARED_S16_N2 | ShaderVariableVec4Info::DECLARED_S16_N1;
                    or_decl |= ShaderVariableVec4Info::DECLARED_S16;
                }

                if (and_decl != 0 || or_decl != 0) {
                    restore_infos.push({ info, odd_index, info.dirty[odd_index], var_key, (info.declared & ~and_decl) | or_decl });
                    info.dirty[odd_index] = -1;
                    info.dirty[odd_index + 1] = -1;
                    break;
                }
            }

            if (has_low) {
                if ((info.dirty[odd_index] != type_dirty) && (info.dirty[odd_index] != -1)) {
                    restore_infos.push({ info, odd_index, info.dirty[odd_index], var_key, info.declared });
                    info.dirty[odd_index] = -1;
                }
            }
            if (has_high) {
                ShaderVariableVec4Info &info_latter = ((odd_index + 1) >= 4) ? info_next : info;
                if ((info_latter.dirty[(odd_index + 1) % 4] != type_dirty) && (info_latter.dirty[(odd_index + 1) % 4] != -1)) {
                    restore_infos.push({ info_latter, odd_index + 1, info_latter.dirty[(odd_index + 1) % 4], var_key, info_latter.declared });
                    info_latter.dirty[(odd_index + 1) % 4] = -1;
                }
            }
            break;
        }

        case 4: {
            for (std::uint8_t i = 0; i < 4; i++) {
                if ((dest_mask & (1 << i)) && (swizz[i] <= SwizzleChannel::C_W)) {
                    int swizz_index = static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X);

                    ShaderVariableVec4Info &info_latter = ((odd_index + swizz_index) >= 4) ? info_next : info;
                    int modded = (odd_index + swizz_index) % 4;

                    if ((info_latter.dirty[modded] != type_dirty) && (info_latter.dirty[modded] != -1)) {
                        restore_infos.push({ info_latter, odd_index + swizz_index, info_latter.dirty[modded], var_key, info_latter.declared });
                        info_latter.dirty[modded] = -1;
                    }
                }
            }
            break;
        }

        default:
            break;
        }

        do_restore(restore_infos, target_bank.prefix);

        // To be truthful I made those note because it got too messy, my brain cant think no more
        if (for_write) {
            if (get_data_type_size(dt) == 1) {
                info.dirty[shifted_load_index % 4] = ((dt == DataType::UINT8) ? ShaderVariableVec4Info::DIRTY_TYPE_U8 : ShaderVariableVec4Info::DIRTY_TYPE_S8);
            } else {
                int mod = shifted_load_index % 4;
                for (std::uint8_t i = 0; i < 4; i++) {
                    if (dest_mask & (1 << i)) {
                        if (swizz[i] <= SwizzleChannel::C_W) {
                            int index = static_cast<int>(swizz[i]) - static_cast<int>(SwizzleChannel::C_X);

                            switch (get_data_type_size(dt)) {
                            case 2: {
                                ShaderVariableVec4Info &info_latter = ((mod + (index / 2)) >= 4) ? target_bank.variables[var_key + 4] : info;
                                switch (dt) {
                                case DataType::UINT16:
                                    info_latter.dirty[(shifted_load_index + (index / 2)) % 4] = ShaderVariableVec4Info::DIRTY_TYPE_U16;
                                    break;

                                case DataType::INT16:
                                    info_latter.dirty[(shifted_load_index + (index / 2)) % 4] = ShaderVariableVec4Info::DIRTY_TYPE_S16;
                                    break;

                                case DataType::F16:
                                    info_latter.dirty[(shifted_load_index + (index / 2)) % 4] = ShaderVariableVec4Info::DIRTY_TYPE_F16;
                                    break;
                                }
                                break;
                            }

                            case 4: {
                                ShaderVariableVec4Info &info_latter = ((mod + index) >= 4) ? target_bank.variables[var_key + 4] : info;
                                info_latter.dirty[(shifted_load_index + index) % 4] = ShaderVariableVec4Info::DIRTY_TYPE_F32;
                                break;
                            }

                            default:
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

static inline std::string format_float(float value) {
    std::string result = fmt::format("{}", value);
    if (result.find(".") == std::string::npos) {
        // Must insert for the compiler to recognise as float.
        // NOTE: There's no way to force fmt to do this, so we need to do it ourselves.
        result += ".0";
    }
    return result;
}

std::string ShaderVariables::load(const Operand &op, const std::uint8_t dest_mask, const int shift_offset) {
    int comp_count = dest_mask_to_comp_count(dest_mask);

    if (op.bank == RegisterBank::FPCONSTANT) {
        const bool integral_unsigned = (op.type == DataType::UINT32) || (op.type == DataType::UINT16);
        const bool integral_signed = (op.type == DataType::INT32) || (op.type == DataType::INT16);
        std::vector<std::string> consts;

        auto handle_unexpect_swizzle = [&](const SwizzleChannel ch) -> std::string {
#define GEN_CONSTANT(cnst)                                                \
    if (integral_unsigned)                                                \
        return fmt::format("{}u", cnst);                                  \
    else if (integral_signed)                                             \
        return fmt::format("{}", cnst);                                   \
    else                                                                  \
        return fmt::format("{}.0", cnst);
            switch (ch) {
            case SwizzleChannel::C_0:
                GEN_CONSTANT(0);
                break;
            case SwizzleChannel::C_1:
                GEN_CONSTANT(1);
                break;
            case SwizzleChannel::C_2:
                GEN_CONSTANT(2);
                break;
            case SwizzleChannel::C_H:
                GEN_CONSTANT(0.5f);
                break;
            default: break;
            }

            return "ERROR";

#undef GEN_CONSTANT
        };

        // Load constants. Ignore mask
        if ((op.type == DataType::F32) || (op.type == DataType::UINT32) || (op.type == DataType::INT32)) {
            auto get_f32_from_bank = [&](const int num) -> std::string {
                int swizz_val = static_cast<int>(op.swizzle[num]) - static_cast<int>(SwizzleChannel::C_X);
                std::uint32_t value = 0;

                switch (swizz_val >> 1) {
                case 0: {
                    value = usse::f32_constant_table_bank_0_raw[op.num + (swizz_val & 1)];
                    break;
                }

                case 1: {
                    value = usse::f32_constant_table_bank_1_raw[op.num + (swizz_val & 1)];
                    break;
                }

                default:
                    return handle_unexpect_swizzle(op.swizzle[num]);
                }

                if (integral_unsigned)
                    return fmt::format("{}u", value);
                else if (integral_signed)
                    return fmt::format("{}", value);
                else
                    return format_float(std::bit_cast<float>(value));
            };

            for (int i = 0; i < 4; i++) {
                if (dest_mask & (1 << i)) {
                    consts.push_back(get_f32_from_bank(i));
                }
            }
        } else if ((op.type == DataType::F16) || (op.type == DataType::UINT16)) {
            auto get_f16_from_bank = [&](const int num) -> std::string {
                const int swizz_val = static_cast<int>(op.swizzle[num]) - static_cast<int>(SwizzleChannel::C_X);
                float value = 0;

                switch (swizz_val & 3) {
                case 0: {
                    value = usse::f16_constant_table_bank0[op.num];
                    break;
                }

                case 1: {
                    value = usse::f16_constant_table_bank1[op.num];
                    break;
                }

                case 2: {
                    value = usse::f16_constant_table_bank2[op.num];
                    break;
                }

                case 3: {
                    value = usse::f16_constant_table_bank3[op.num];
                    break;
                }

                default:
                    return handle_unexpect_swizzle(op.swizzle[num]);
                }

                if (integral_unsigned)
                    return fmt::format("{}u", std::bit_cast<uint32_t>(value));
                else if (integral_signed)
                    return fmt::format("{}", std::bit_cast<int32_t>(value));
                else
                    return format_float(value);
            };

            for (int i = 0; i < 4; i++) {
                if (dest_mask & (1 << i)) {
                    consts.push_back(get_f16_from_bank(i));
                }
            }
        }

        std::string final_string = "";
        if (consts.size() == 1) {
            final_string = consts[0];
        } else {
            final_string = is_unsigned_integer_data_type(op.type) ? "uvec" : (is_signed_integer_data_type(op.type) ? "ivec" : "vec");
            final_string += static_cast<char>('0' + consts.size());
            final_string += "(";

            for (std::size_t i = 0; i < consts.size(); i++) {
                final_string += consts[i] + ", ";
            }

            final_string.pop_back();
            final_string.back() = ')';
        }

        apply_modifiers(op.flags, final_string);
        return final_string;
    }

    if (op.bank == RegisterBank::IMMEDIATE) {
        std::string var_comp;
        std::string imm = fmt::format("{}", op.num);
        std::string postfix;

        if (is_unsigned_integer_data_type(op.type)) {
            postfix = "u";
        } else if (is_float_data_type(op.type)) {
            postfix = ".0";
        }

        imm += postfix;
        std::uint8_t total_comp = 0;

        for (std::uint8_t i = 0; i < dest_mask; i++) {
            if (dest_mask & (1 << i)) {
                total_comp++;
                if (op.swizzle[i] <= SwizzleChannel::C_W) {
                    var_comp += imm + ", ";
                } else {
                    switch (op.swizzle[i]) {
                    case SwizzleChannel::C_0:
                        var_comp += std::string("0") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_1:
                        var_comp += std::string("1") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_2:
                        var_comp += std::string("2") + postfix + ", ";
                        break;

                    case SwizzleChannel::C_H:
                        var_comp += "0.5, ";
                        break;
                    }
                }
            }
        }
        var_comp.pop_back();
        var_comp.pop_back();
        if (total_comp <= 1) {
            return var_comp;
        }
        return fmt::format("{}vec{}({})", is_signed_integer_data_type(op.type) ? "i" : (is_unsigned_integer_data_type(op.type) ? "u" : ""), total_comp,
            var_comp);
    }

    if (op.bank == RegisterBank::INDEX) {
        if (!index_variables[op.num - 1]) {
            writer.add_declaration(fmt::format("int index{};", op.num));
            index_variables[op.num - 1] = true;
        }

        return fmt::format("index{}\n", op.num);
    }

    std::string source = "";

    if ((op.bank == RegisterBank::INDEXED1) || (op.bank == RegisterBank::INDEXED2)) {
        // It's a rare case, but we hack a bit, usually they should have only used this if the offset is already aligned by 4
        // Only some shaders in games like Downwell use this. Most of the time they will just use VLDR/VSTR
        // Since the infustructure here discourages big array indexing, we will do a switch of possibly known SA variables,
        // include ones that are already accessed in the shaders and one that in uniform banks (if SA)
        const Imm2 bank_enc = (op.num >> 5) & 0b11;
        const Imm5 add_off = (op.num & 0b11111) + shift_offset;

        source = "tryIndex";

        switch (bank_enc) {
        case 0: {
            source += "TempBank";
            break;
        }

        case 1: {
            source += "OutputBank";
            break;
        }

        case 2: {
            source += "PaBank";
            break;
        }

        case 3: {
            source += "SaBank";
            break;
        }

        default:
            break;
        }

        // NOTE: As stated before, we will just access these with the assumption its already aligned by 4
        source += fmt::format("((index{} * 2 + {}) / 4 * 4)", (op.bank == RegisterBank::INDEXED1) ? 1 : 2, add_off);
        should_gen_indexing_lookup[bank_enc] = true;
    } else {
        // Resume normal operation
        DataType real_data_type = op.type;

        int shifted_load_index = op.num + shift_offset;
        int real_load_index = shifted_load_index;

        bool load_to_another_vec = false;
        std::string var_name;

        ShaderVariableBank &target_bank = *get_reg_bank(op.bank);
        prepare_variables(target_bank, real_data_type, op.bank, op.num, shift_offset, op.swizzle,
            dest_mask, false, var_name, real_load_index, load_to_another_vec);

        if (op.bank == RegisterBank::FPINTERNAL) {
            if (is_float_data_type(real_data_type)) {
                real_data_type = DataType::F32;
            } else if (is_unsigned_integer_data_type(real_data_type)) {
                real_data_type = DataType::UINT32;
            } else {
                real_data_type = DataType::INT32;
            }

            var_name = generate_read_vector(var_name, real_load_index, op.swizzle, dest_mask, real_data_type, true);
        } else {
            var_name = generate_read_vector(var_name, real_load_index, op.swizzle, dest_mask, real_data_type);
        }

        if ((real_data_type != DataType::UINT8) && (real_data_type != DataType::INT8)) {
            switch (real_data_type) {
            case DataType::UINT32:
                var_name = std::string("floatBitsToUint(") + var_name + ")";
                break;

            case DataType::INT32:
                var_name = std::string("floatBitsToInt(") + var_name + ")";
                break;

            default:
                break;
            }
        }

        source = var_name;
    }

    apply_modifiers(op.flags, source);
    return source;
}

void ShaderVariables::apply_modifiers(const std::uint32_t flags, std::string &result) {
    if (flags & RegisterFlags::Absolute) {
        result = std::string("abs(") + result + ")";
    }

    if (flags & RegisterFlags::Negative) {
        result = std::string("-") + result;
    }
}

void ShaderVariables::store(const Operand &dest, const std::string &rhs, const std::uint8_t dest_mask, const int shift_offset, const bool prepare_store) {
    if (dest.bank == RegisterBank::INDEX) {
        if (!index_variables[dest.num - 1]) {
            writer.add_declaration(fmt::format("int index{};", dest.num));
            index_variables[dest.num - 1] = true;
        }

        writer.add_to_current_body(fmt::format("index{} = {};", dest.num, rhs));
        return;
    }

    if ((dest.bank == RegisterBank::INDEXED1) || (dest.bank == RegisterBank::INDEXED2)) {
        LOG_TRACE("Storing through indexed not supported! If you see this contact developers!");
        return;
    }

    DataType real_data_type = dest.type;
    int real_load_index = dest.num + shift_offset;
    std::string var_name;
    bool store_to_another_vec = false;

    ShaderVariableBank &target_bank = *get_reg_bank(dest.bank);
    var_name = target_bank.prefix;

    // We still need to preload data for safe case. In future, we should avoid doing this by check if it will load
    // the whole full data
    prepare_variables(target_bank, real_data_type, dest.bank, dest.num, shift_offset, SWIZZLE_CHANNEL_4_DEFAULT,
        dest_mask, true, var_name, real_load_index, store_to_another_vec);

    std::string rhs_modded = rhs;

    // Prepare the RHS
    switch (dest.type) {
    case DataType::F16:
        should_gen_clamp16 = true;
        rhs_modded = std::string("clampf16(") + rhs + ")";
        break;

    case DataType::UINT8:
        should_gen_vecfloat_bitcast[0] = true;
        rhs_modded = std::string("clampU8(") + rhs + ")";
        break;

    case DataType::INT8:
        should_gen_vecfloat_bitcast[1] = true;
        rhs_modded = std::string("clampS8(") + rhs + ")";
        break;

    case DataType::UINT16:
        should_gen_vecfloat_bitcast[2] = true;
        rhs_modded = std::string("clampU16(") + rhs + ")";
        break;

    case DataType::INT16:
        should_gen_vecfloat_bitcast[3] = true;
        rhs_modded = std::string("clampS16(") + rhs + ")";
        break;

    case DataType::UINT32:
        rhs_modded = std::string("uintBitsToFloat(") + rhs + ")";
        break;

    case DataType::INT32:
        rhs_modded = std::string("intBitsToFloat(") + rhs + ")";
        break;

    default:
        break;
    }

    // If aligned or just FPINTERNAL
    if ((dest.bank == RegisterBank::FPINTERNAL) || (real_load_index % 4 == 0)) {
        std::string swizz_str;
        for (std::uint8_t i = 0; i < 4; i++) {
            if (dest_mask & (1 << i)) {
                swizz_str += static_cast<char>('w' + (i + 1) % 4);
            }
        }

        if (dest.bank == RegisterBank::FPINTERNAL) {
            // After clamp done we can cast
            if (is_signed_integer_data_type(dest.type)) {
                rhs_modded = std::string("intBitsToFloat(") + rhs + ")";
            } else if (is_unsigned_integer_data_type(dest.type)) {
                rhs_modded = std::string("uintBitsToFloat(") + rhs + ")";
            }
        }

        if (dest_mask == 0b1111) {
            writer.add_to_current_body(fmt::format("{}{} = {};", var_name, real_load_index, rhs_modded));
        } else {
            writer.add_to_current_body(fmt::format("{}{}.{} = {};", var_name, real_load_index, swizz_str, rhs_modded));
        }
    } else {
        // We may need to divide it out. But first we need to know if it only uses the next vec4?
        int mod = (real_load_index % 4);
        bool only_use_next_vec4 = true;
        bool only_use_this_vec4 = true;

        for (std::uint8_t i = 0; i < 4; i++) {
            if (dest_mask & (1 << i)) {
                if (only_use_next_vec4 && ((mod + i) < 4)) {
                    only_use_next_vec4 = false;
                }

                if (only_use_this_vec4 && (mod + i) >= 4) {
                    only_use_this_vec4 = false;
                }
            }
        }

        if (only_use_next_vec4 || only_use_this_vec4) {
            if (mod == 0 && dest_mask == 0b1111) {
                writer.add_to_current_body(fmt::format("{}{} = {};", var_name, only_use_this_vec4 ? (real_load_index / 4) * 4 : (real_load_index / 4 + 1) * 4, rhs_modded));
            } else {
                std::string swizz_str;
                for (std::uint8_t i = 0; i < 4; i++) {
                    if (dest_mask & (1 << i)) {
                        swizz_str += static_cast<char>('w' + ((i + mod + 1) % 4));
                    }
                }

                writer.add_to_current_body(fmt::format("{}{}.{} = {};", var_name, only_use_this_vec4 ? (real_load_index / 4) * 4 : (real_load_index / 4 + 1) * 4, swizz_str, rhs_modded));
            }
        } else {
            int num_comp_count = dest_mask_to_comp_count(dest_mask);
            std::string temp_name = fmt::format("temp{}", num_comp_count);

            // We must separate it out
            writer.add_to_current_body(fmt::format("{} = {};", temp_name, rhs_modded));
            std::string swizz_str;
            std::string swizz_str_full = "xyzw";

            for (std::uint8_t i = 0; i < 4 - mod; i++) {
                if (dest_mask & (1 << i)) {
                    swizz_str += static_cast<char>('w' + (i + mod + 1) % 4);
                }
            }

            writer.add_to_current_body(fmt::format("{}{}.{} = {}.{};", var_name, real_load_index / 4 * 4, swizz_str, temp_name, swizz_str_full.substr(0, 4 - mod)));

            swizz_str = "";
            for (std::uint8_t i = 4 - mod; i < 4; i++) {
                if (dest_mask & (1 << i)) {
                    swizz_str += static_cast<char>('w' + ((i + mod + 1) % 4));
                }
            }

            writer.add_to_current_body(fmt::format("{}{}.{} = {}.{};", var_name, (real_load_index / 4 + 1) * 4, swizz_str, temp_name, swizz_str_full.substr(4 - mod, mod - (4 - num_comp_count))));
        }
    }
}

void ShaderVariables::generate_helper_functions() {
    if (should_gen_clamp16) {
        writer.add_declaration("vec2 clampf16(vec2 value) { return unpackHalf2x16(packHalf2x16(value)); }");
        writer.add_declaration("float clampf16(float value) { return clampf16(vec2(value, 0.0)).x; }");
        writer.add_declaration("vec3 clampf16(vec3 value) { return vec3(clampf16(value.xy), clampf16(value.z)); }");
        writer.add_declaration("vec4 clampf16(vec4 value) { return vec4(clampf16(value.xy), clampf16(value.zw)); }\n");
    }

    if (should_gen_vecfloat_bitcast[0]) {
        writer.add_declaration("int clampU8(uint value) { return int(clamp(value, 0u, 255u)); }");
        writer.add_declaration("ivec2 clampU8(uvec2 value) { return ivec2(clamp(value, uvec2(0), uvec2(255))); }");
        writer.add_declaration("ivec3 clampU8(uvec3 value) { return ivec3(clamp(value, uvec3(0), uvec3(255))); }");
        writer.add_declaration("ivec4 clampU8(uvec4 value) { return ivec4(clamp(value, uvec4(0), uvec4(255))); }");
        writer.add_declaration("int clampU8(int value) { return clamp(value, 0, 255); }");
        writer.add_declaration("ivec2 clampU8(ivec2 value) { return clamp(value, ivec2(0), ivec2(255)); }");
        writer.add_declaration("ivec3 clampU8(ivec3 value) { return clamp(value, ivec3(0), ivec3(255)); }");
        writer.add_declaration("ivec4 clampU8(ivec4 value) { return clamp(value, ivec4(0), ivec4(255)); }");
        writer.add_declaration("int clampU8(float value) { return clamp(int(value), 0, 255); }");
        writer.add_declaration("ivec2 clampU8(vec2 value) { return clamp(ivec2(value), ivec2(0), ivec2(255)); }");
        writer.add_declaration("ivec3 clampU8(vec3 value) { return clamp(ivec3(value), ivec3(0), ivec3(255)); }");
        writer.add_declaration("ivec4 clampU8(vec4 value) { return clamp(ivec4(value), ivec4(0), ivec4(255)); }\n");
    }

    if (should_gen_vecfloat_bitcast[1]) {
        writer.add_declaration("int clampS8(int value) { return clamp(value, -128, 127); }");
        writer.add_declaration("ivec2 clampS8(ivec2 value) { return clamp(value, ivec2(-128), ivec2(127)); }");
        writer.add_declaration("ivec3 clampS8(ivec3 value) { return clamp(value, ivec3(-128), ivec3(127)); }");
        writer.add_declaration("ivec4 clampS8(ivec4 value) { return clamp(value, ivec4(-128), ivec4(127)); }");
        writer.add_declaration("int clampS8(uint value) { return int(clamp(value, 0u, 127u)); }");
        writer.add_declaration("ivec2 clampS8(uvec2 value) { return ivec2(clamp(value, uvec2(0), uvec2(127))); }");
        writer.add_declaration("ivec3 clampS8(uvec3 value) { return ivec3(clamp(value, uvec3(0), uvec3(127))); }");
        writer.add_declaration("ivec4 clampS8(uvec4 value) { return ivec4(clamp(value, uvec4(0), uvec4(127))); }");
        writer.add_declaration("int clampS8(float value) { return clamp(int(value), -128, 127); }");
        writer.add_declaration("ivec2 clampS8(vec2 value) { return clamp(ivec2(value), ivec2(-128), ivec2(127)); }");
        writer.add_declaration("ivec3 clampS8(vec3 value) { return clamp(ivec3(value), ivec3(-128), ivec3(127)); }");
        writer.add_declaration("ivec4 clampS8(vec4 value) { return clamp(ivec4(value), ivec4(-128), ivec4(127)); }\n");
    }

    if (should_gen_vecfloat_bitcast[2]) {
        writer.add_declaration("int clampU16(uint value) { return int(clamp(value, 0u, 65535u)); }");
        writer.add_declaration("ivec2 clampU16(uvec2 value) { return ivec2(clamp(value, uvec2(0), uvec2(65535))); }");
        writer.add_declaration("ivec3 clampU16(uvec3 value) { return ivec3(clamp(value, uvec3(0), uvec3(65535))); }");
        writer.add_declaration("ivec4 clampU16(uvec4 value) { return ivec4(clamp(value, uvec4(0), uvec4(65535))); }");
        writer.add_declaration("int clampU16(int value) { return clamp(value, 0, 65535); }");
        writer.add_declaration("ivec2 clampU16(ivec2 value) { return clamp(value, ivec2(0), ivec2(65535)); }");
        writer.add_declaration("ivec3 clampU16(ivec3 value) { return clamp(value, ivec3(0), ivec3(65535)); }");
        writer.add_declaration("ivec4 clampU16(ivec4 value) { return clamp(value, ivec4(0), ivec4(65535)); }");
        writer.add_declaration("int clampU16(float value) { return clamp(int(value), 0, 65535); }");
        writer.add_declaration("ivec2 clampU16(vec2 value) { return clamp(ivec2(value), ivec2(0), ivec2(65535)); }");
        writer.add_declaration("ivec3 clampU16(vec3 value) { return clamp(ivec3(value), ivec3(0), ivec3(65535)); }");
        writer.add_declaration("ivec4 clampU16(vec4 value) { return clamp(ivec4(value), ivec4(0), ivec4(65535)); }\n");
    }

    if (should_gen_vecfloat_bitcast[3]) {
        writer.add_declaration("int clampS16(int value) { return clamp(value, -32768, 32767); }");
        writer.add_declaration("ivec2 clampS16(ivec2 value) { return clamp(value, ivec2(-32768), ivec2(32767)); }");
        writer.add_declaration("ivec3 clampS16(ivec3 value) { return clamp(value, ivec3(-32768), ivec3(32767)); }");
        writer.add_declaration("ivec4 clampS16(ivec4 value) { return clamp(value, ivec4(-32768), ivec4(32767)); }");
        writer.add_declaration("int clampS16(uint value) { return int(clamp(value, 0u, 32767u)); }");
        writer.add_declaration("ivec2 clampS16(uvec2 value) { return ivec2(clamp(value, uvec2(0), uvec2(32767))); }");
        writer.add_declaration("ivec3 clampS16(uvec3 value) { return ivec3(clamp(value, uvec3(0), uvec3(32767))); }");
        writer.add_declaration("ivec4 clampS16(uvec4 value) { return ivec4(clamp(value, uvec4(0), uvec4(32767))); }");
        writer.add_declaration("int clampS16(float value) { return clamp(int(value), -32768, 32767); }");
        writer.add_declaration("ivec2 clampS16(vec2 value) { return clamp(ivec2(value), ivec2(-32768), ivec2(32767)); }");
        writer.add_declaration("ivec3 clampS16(vec3 value) { return clamp(ivec3(value), ivec3(-32768), ivec3(32767)); }");
        writer.add_declaration("ivec4 clampS16(vec4 value) { return clamp(ivec4(value), ivec4(-32768), ivec4(32767)); }\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_4XU8]) {
        writer.add_declaration("float pack4xU8(uvec4 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFFu;");
        writer.add_declaration("uint result = bitfieldInsert(bitfieldInsert(bitfieldInsert(val.x, val.y, 8, 8), val.z, 16, 8), val.w, 24, 8);");
        writer.add_declaration("return uintBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_4XS8]) {
        writer.add_declaration("float pack4xS8(ivec4 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFF;");
        writer.add_declaration("int result = bitfieldInsert(bitfieldInsert(bitfieldInsert(val.x, val.y, 8, 8), val.z, 16, 8), val.w, 24, 8);");
        writer.add_declaration("return intBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_4XU8]) {
        writer.add_declaration("uvec4 unpack4xU8(float val) {");
        writer.indent_declaration();
        writer.add_declaration("uint cc = floatBitsToUint(val);");
        writer.add_declaration("return uvec4(bitfieldExtract(cc, 0, 8), bitfieldExtract(cc, 8, 8), bitfieldExtract(cc, 16, 8), bitfieldExtract(cc, 24, 8));");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_4XS8]) {
        writer.add_declaration("ivec4 unpack4xS8(float val) {");
        writer.indent_declaration();
        writer.add_declaration("int cc = floatBitsToInt(val);");
        writer.add_declaration("return ivec4(bitfieldExtract(cc, 0, 8), bitfieldExtract(cc, 8, 8), bitfieldExtract(cc, 16, 8), bitfieldExtract(cc, 24, 8));");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_2XF16]) {
        writer.add_declaration("float pack2xF16(vec2 val) { return uintBitsToFloat(packHalf2x16(val)); }\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_4XF16]) {
        writer.add_declaration("vec2 pack4xF16(vec4 val) { return vec2(uintBitsToFloat(packHalf2x16(val.xy)), uintBitsToFloat(packHalf2x16(val.zw))); }\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_2XF16]) {
        writer.add_declaration("vec2 unpack2xF16(float val) { return unpackHalf2x16(floatBitsToUint(val)); }\n");
    }
    
    if (should_gen_pack_unpack[GEN_UNPACK_4XF16]) {
        writer.add_declaration("vec4 unpack4xF16(vec2 val) { return vec4(unpackHalf2x16(floatBitsToUint(val.x)), unpackHalf2x16(floatBitsToUint(val.y))); }\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_2XU16]) {
        writer.add_declaration("float pack2xU16(uvec2 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFFFFu;");
        writer.add_declaration("uint result = bitfieldInsert(val.x, val.y, 16, 16);");
        writer.add_declaration("return uintBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_4XU16]) {
        writer.add_declaration("vec2 pack4xU16(uvec4 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFFFFu;");
        writer.add_declaration("uvec2 result;");
        writer.add_declaration("result.x = bitfieldInsert(val.x, val.y, 16, 16);");
        writer.add_declaration("result.y = bitfieldInsert(val.z, val.w, 16, 16);");
        writer.add_declaration("return uintBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_2XS16]) {
        writer.add_declaration("float pack2xS16(ivec2 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFFFF;");
        writer.add_declaration("int result = bitfieldInsert(val.x, val.y, 16, 16);");
        writer.add_declaration("return intBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_PACK_4XS16]) {
        writer.add_declaration("vec2 pack4xS16(ivec4 val) {");
        writer.indent_declaration();
        writer.add_declaration("val &= 0xFFFF;");
        writer.add_declaration("ivec2 result;");
        writer.add_declaration("result.x = bitfieldInsert(val.x, val.y, 16, 16);");
        writer.add_declaration("result.y = bitfieldInsert(val.z, val.w, 16, 16);");
        writer.add_declaration("return intBitsToFloat(result);");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_2XU16]) {
        writer.add_declaration("uvec2 unpack2xU16(float val) {");
        writer.indent_declaration();
        writer.add_declaration("uint cc = floatBitsToUint(val);");
        writer.add_declaration("return uvec2(bitfieldExtract(cc, 0, 16), bitfieldExtract(cc, 16, 16));");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }
        
    if (should_gen_pack_unpack[GEN_UNPACK_4XU16]) {
        writer.add_declaration("uvec4 unpack4xU16(vec2 val) {");
        writer.indent_declaration();
        writer.add_declaration("uvec2 cc = floatBitsToUint(val);");
        writer.add_declaration("uvec4 result;");
        writer.add_declaration("result.xy = uvec2(bitfieldExtract(cc.x, 0, 16), bitfieldExtract(cc.x, 16, 16));");
        writer.add_declaration("result.zw = uvec2(bitfieldExtract(cc.y, 0, 16), bitfieldExtract(cc.y, 16, 16));");
        writer.add_declaration("return result;");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_2XS16]) {
        writer.add_declaration("ivec2 unpack2xS16(float val) {");
        writer.indent_declaration();
        writer.add_declaration("int cc = floatBitsToInt(val);");
        writer.add_declaration("return ivec2(bitfieldExtract(cc, 0, 16), bitfieldExtract(cc, 16, 16));");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_4XS16]) {
        writer.add_declaration("ivec4 unpack4xS16(vec2 val) {");
        writer.indent_declaration();
        writer.add_declaration("ivec2 cc = floatBitsToInt(val);");
        writer.add_declaration("ivec4 result;");
        writer.add_declaration("result.xy = ivec2(bitfieldExtract(cc.x, 0, 16), bitfieldExtract(cc.x, 16, 16));");
        writer.add_declaration("result.zw = ivec2(bitfieldExtract(cc.y, 0, 16), bitfieldExtract(cc.y, 16, 16));");
        writer.add_declaration("return result;");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_pack_unpack[GEN_UNPACK_3XFX10]) {
        writer.add_declaration("ivec4 unpack3xFX10(float val) {");
        writer.indent_declaration();
        writer.add_declaration("int cc = floatBitsToInt(val);");
        writer.add_declaration("return ivec3(bitfieldExtract(cc, 0, 10), bitfieldExtract(cc, 10, 10), bitfieldExtract(cc, 20, 10));");
        writer.dedent_declaration();
        writer.add_declaration("}\n");
    }

    if (should_gen_textureprojcube) {
        writer.add_declaration("vec4 textureProjCube(samplerCube sampler, vec4 coord) {");
        writer.indent_declaration();
        writer.add_declaration("return texture(sampler, coord.xyz / coord.w);");
        writer.dedent_declaration();
        writer.add_declaration("}");
    }
    
    if (should_gen_texture_get_bilinear_coefficients) {
        writer.add_declaration("vec4 textureGetBilinearCoefficients(vec2 uv) {");
        writer.indent_declaration();
        // compute and save bilinear coefficients
        // the pixels returned by gather4 are in the following order : (0,1) (1,1) (1,0) (0,0)
        // so at the end we want (1-u)v uv u(1-v) (1-u)(1-v)
        // however, looking at the generated shader code in some games, it looks like the coefficients are
        // expected to be in this order but reversed...

        // (1-u) u
        const std::string x_coeffs = "vec2(1.0 - uv.x, uv.x)";
        // (1-u)v uv
        const std::string comp1 = x_coeffs + " * uv.y";
        // (1-u)(1-v) u(1-v)
        const std::string comp2 = x_coeffs + " * (1.0 - uv.y)";
        // (1-u)v uv u(1-v) (1-u)(1-v) in reversed order
        const std::string coeffs = std::string("vec4(") + comp1 + ", " + comp2 + ").zwyx";

        writer.add_declaration(std::string("return ") + coeffs + ";");
        writer.dedent_declaration();
        writer.add_declaration("}");
    }
}

static float get_int_normalize_range_constants(DataType type) {
    switch (type) {
    case DataType::UINT8:
        return 255.0f;
    case DataType::INT8:
        return 127.0f;
    case DataType::C10:
        // signed 10-bit, with a range of [-2, 2]
        return 255.0f;
    case DataType::UINT16:
        return 65535.0f;
    case DataType::INT16:
        return 32767.0f;
    case DataType::UINT32:
        return 4294967295.0f;
    case DataType::INT32:
        return 2147483647.0f;
    default:
        assert(false);
        return 0.0f;
    }
}

std::string ShaderVariables::convert_to_float(std::string src, const std::int32_t comp_count, DataType type, bool normal) {
    if (is_float_data_type(type)) {
        return src;
    }

    if (normal) {
        const float constant_range = get_int_normalize_range_constants(type);
        const std::string normalizer_str = constant_range != 0.0f ? fmt::format(" / {}", format_float(constant_range)) : "";

        if (comp_count == 1) {
            src = fmt::format("float({}){}", src, normalizer_str);
        } else {
            src = fmt::format("vec{}({}){}", comp_count, src, normalizer_str);
        }
    } else {
        if (comp_count == 1) {
            src = fmt::format("float({})", src);
        } else {
            src = fmt::format("vec{}({})", comp_count, src);
        }
    }

    return src;
}

std::string ShaderVariables::convert_to_int(std::string src, const std::int32_t comp_count, DataType type, bool normal) {
    if (!is_float_data_type(type)) {
        return src;
    }

    const std::string dest_dt_type = is_unsigned_integer_data_type(type) ? "uint" : "int";
    if (normal) {
        const float constant_range = get_int_normalize_range_constants(type);
        const std::string normalizer_str = constant_range != 0.0f ? fmt::format(" * {}", format_float(constant_range)) : "";

        const bool is_uint = is_unsigned_integer_data_type(type);
        const bool is_fx10 = type == DataType::C10; // fx10 range is [-2,2]
        const std::string range_begin = format_float(is_uint ? 0.0 : (is_fx10 ? -2.0 : -1.0));
        const std::string range_end = format_float(is_fx10 ? 2.0 : 1.0);

        if (comp_count == 1) {
            src = fmt::format("{}(round(clamp({}, {}, {}){}))",
                dest_dt_type, src, range_begin, range_end, normalizer_str);
        } else {
            src = fmt::format("{}vec{}(round(clamp({}, vec{}({}), vec{}({})){}))",
                dest_dt_type[0], comp_count, src, comp_count, range_begin, comp_count, range_end, normalizer_str);
        }
    } else {
        if (comp_count == 1) {
            src = fmt::format("{}({})", dest_dt_type, src);
        } else {
            src = fmt::format("{}vec{}({})", dest_dt_type[0], comp_count, src);
        }
    }

    return src;
}
} // namespace shader::usse::glsl