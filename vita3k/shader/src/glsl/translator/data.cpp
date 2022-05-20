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

#include <shader/decoder_helpers.h>
#include <shader/disasm.h>
#include <shader/glsl/code_writer.h>
#include <shader/glsl/params.h>
#include <shader/glsl/translator.h>
#include <shader/types.h>
#include <util/log.h>

using namespace shader;
using namespace usse;
using namespace glsl;

bool USSETranslatorVisitorGLSL::vmov(
    ExtPredicate pred,
    bool skipinv,
    Imm1 test_bit_2,
    Imm1 src0_comp_sel,
    bool syncstart,
    Imm1 dest_bank_ext,
    Imm1 end_or_src0_bank_ext,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    MoveType move_type,
    RepeatCount repeat_count,
    bool nosched,
    DataType move_data_type,
    Imm1 test_bit_1,
    Imm4 src0_swiz,
    Imm1 src0_bank_sel,
    Imm2 dest_bank_sel,
    Imm2 src1_bank_sel,
    Imm2 src2_bank_sel,
    Imm4 dest_mask,
    Imm6 dest_n,
    Imm6 src0_n,
    Imm6 src1_n,
    Imm6 src2_n) {
    if (!USSETranslatorVisitor::vmov(pred, skipinv, test_bit_2, src0_comp_sel, syncstart, dest_bank_ext, end_or_src0_bank_ext,
            src1_bank_ext, src2_bank_ext, move_type, repeat_count, nosched, move_data_type, test_bit_1, src0_swiz,
            src0_bank_sel, dest_bank_sel, src1_bank_sel, src2_bank_sel, dest_mask, dest_n, src0_n, src1_n, src2_n)) {
        return false;
    }

    const bool is_conditional = (move_type != MoveType::UNCONDITIONAL);
    const bool is_u8_conditional = decoded_inst.opcode == Opcode::VMOVCU8;
    const std::size_t comp_count = dest_mask_to_comp_count(decoded_dest_mask);

    std::string compare_func;

    if (is_conditional) {
        CompareMethod compare_method = static_cast<CompareMethod>((test_bit_2 << 1) | test_bit_1);

        switch (compare_method) {
        case CompareMethod::LT_ZERO:
            compare_func = ((comp_count > 1) ? "lessThan" : "<");
            break;
        case CompareMethod::LTE_ZERO:
            compare_func = ((comp_count > 1) ? "lessThanEqual" : "<=");
            break;
        case CompareMethod::NE_ZERO:
            compare_func = ((comp_count > 1) ? "notEqual" : "!=");
            break;
        case CompareMethod::EQ_ZERO:
            compare_func = ((comp_count > 1) ? "equal" : "<");
            break;
        }
    }

    if ((move_data_type == DataType::F16) || (move_data_type == DataType::F32)) {
        set_repeat_multiplier(2, 2, 2, 2);
    } else {
        set_repeat_multiplier(1, 1, 1, 1);
    }

    BEGIN_REPEAT(repeat_count)
    GET_REPEAT(decoded_inst, RepeatMode::SLMSI);

    std::string source_1 = variables.load(decoded_inst.opr.src1, decoded_dest_mask, src1_repeat_offset);
    std::string result;

    if (source_1.empty()) {
        LOG_ERROR("Source not Loaded");
        return false;
    }

    if (is_conditional) {
        std::string source_to_compare_with_0 = variables.load(decoded_inst.opr.src0, decoded_dest_mask, src0_repeat_offset);
        std::string source_2 = variables.load(decoded_inst.opr.src2, decoded_dest_mask, src2_repeat_offset);

        if (comp_count == 1) {
            result = fmt::format("{} {} 0 ? {} : {}", source_to_compare_with_0, compare_func, source_1, source_2);
        } else {
            if (is_float_data_type(decoded_inst.opr.src1.type) || m_features.support_glsl_mixing_integers) {
                result = fmt::format("mix({}, {}, {}({}, {}vec{}(0)))", source_2, source_1, compare_func, source_to_compare_with_0,
                    is_u8_conditional ? "u" : "", comp_count);
            } else {
                char prefix = (is_unsigned_integer_data_type(decoded_inst.opr.src1.type)) ? 'u' : 'i';
                result = fmt::format("{} * (1{} - {}vec{}({}({}, {}vec{}(0)))) + {} * {}vec{}({}({}, {}vec{}(0)))",
                    source_2, is_unsigned_integer_data_type(decoded_inst.opr.src1.type) ? "u" : "",
                    prefix, comp_count, compare_func, source_to_compare_with_0, prefix, comp_count,
                    source_1, prefix, comp_count, compare_func, source_to_compare_with_0, prefix, comp_count);
            }
        }
    } else {
        result = source_1;
    }

    variables.store(decoded_inst.opr.dest, result, decoded_dest_mask, dest_repeat_offset);
    END_REPEAT()

    reset_repeat_multiplier();
    return true;
}

bool USSETranslatorVisitorGLSL::vpck(
    ExtPredicate pred,
    bool skipinv,
    bool nosched,
    Imm1 unknown,
    bool syncstart,
    Imm1 dest_bank_ext,
    Imm1 end,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    RepeatCount repeat_count,
    Imm3 src_fmt,
    Imm3 dest_fmt,
    Imm4 dest_mask,
    Imm2 dest_bank_sel,
    Imm2 src1_bank_sel,
    Imm2 src2_bank_sel,
    Imm7 dest_n,
    Imm2 comp_sel_3,
    Imm1 scale,
    Imm2 comp_sel_1,
    Imm2 comp_sel_2,
    Imm6 src1_n,
    Imm1 comp0_sel_bit1,
    Imm6 src2_n,
    Imm1 comp_sel_0_bit0) {
    if (!USSETranslatorVisitor::vpck(pred, skipinv, nosched, unknown, syncstart, dest_bank_ext,
            end, src1_bank_ext, src2_bank_ext, repeat_count, src_fmt, dest_fmt, dest_mask, dest_bank_sel,
            src1_bank_sel, src2_bank_sel, dest_n, comp_sel_3, scale, comp_sel_1, comp_sel_2, src1_n,
            comp0_sel_bit1, src2_n, comp_sel_0_bit0)) {
        return false;
    }

    // Doing this extra dest type check for future change in case I'm wrong (pent0)
    if (is_integer_data_type(decoded_inst.opr.dest.type)) {
        if (is_float_data_type(decoded_inst.opr.src1.type)) {
            set_repeat_multiplier(1, 2, 2, 1);
        } else {
            set_repeat_multiplier(1, 1, 1, 1);
        }
    } else {
        if (is_float_data_type(decoded_inst.opr.src1.type)) {
            set_repeat_multiplier(1, 2, 2, 1);
        } else {
            set_repeat_multiplier(1, 1, 1, 1);
        }
    }

    BEGIN_REPEAT(repeat_count)
    GET_REPEAT(decoded_inst, RepeatMode::SLMSI);

    std::string source = variables.load(decoded_inst.opr.src1, dest_mask, src1_repeat_offset);

    if (source.empty()) {
        LOG_ERROR("Source not loaded");
        return false;
    }

    std::string vec_prefix_str = is_unsigned_integer_data_type(decoded_inst.opr.src1.type) ? "u" : (is_signed_integer_data_type(decoded_inst.opr.src1.type) ? "i" : "");

    const std::size_t comp_count = dest_mask_to_comp_count(dest_mask);

    if (decoded_inst.opr.src2.type != DataType::UNK) {
        Operand copy_src1 = decoded_inst.opr.src1;
        Operand copy_src2 = decoded_inst.opr.src2;

        copy_src1.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        copy_src2.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;

        std::string source1 = variables.load(copy_src1, 0b11, src1_repeat_offset);
        std::string source2 = variables.load(copy_src2, 0b11, src2_repeat_offset);

        std::string swizz_str;
        for (std::uint32_t i = 0; i < 4; i++) {
            if (dest_mask & (1 << i)) {
                swizz_str += static_cast<char>('w' + ((static_cast<int>(decoded_inst.opr.src1.swizzle[i]) - static_cast<int>(SwizzleChannel::C_X) + 1) % 4));
            }
        }

        source = fmt::format("{}vec4({}, {}).{}", vec_prefix_str, source1, source2, swizz_str);
    }

    // source is float destination is int
    if (is_float_data_type(decoded_inst.opr.dest.type) && !is_float_data_type(decoded_inst.opr.src1.type)) {
        source = variables.convert_to_float(source, comp_count, decoded_inst.opr.src1.type, scale);
    }

    // source is int destination is float
    if (!is_float_data_type(decoded_inst.opr.dest.type) && is_float_data_type(decoded_inst.opr.src1.type)) {
        source = variables.convert_to_int(source, comp_count, decoded_inst.opr.dest.type, scale);
    }

    variables.store(decoded_inst.opr.dest, source, dest_mask, dest_repeat_offset);
    END_REPEAT()

    reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitorGLSL::vldst(
    Imm2 op1,
    ExtPredicate pred,
    Imm1 skipinv,
    Imm1 nosched,
    Imm1 moe_expand,
    Imm1 sync_start,
    Imm1 cache_ext,
    Imm1 src0_bank_ext,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    Imm4 mask_count,
    Imm2 addr_mode,
    Imm2 mode,
    Imm1 dest_bank_primattr,
    Imm1 range_enable,
    Imm2 data_type,
    Imm1 increment_or_decrement,
    Imm1 src0_bank,
    Imm1 cache_by_pass12,
    Imm1 drc_sel,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    if (!USSETranslatorVisitor::vldst(op1, pred, skipinv, nosched, moe_expand, sync_start, cache_ext, src0_bank_ext,
            src1_bank_ext, src2_bank_ext, mask_count, addr_mode, mode, dest_bank_primattr, range_enable, data_type,
            increment_or_decrement, src0_bank, cache_by_pass12, drc_sel, src1_bank, src2_bank, dest_n, src0_n, src1_n,
            src2_n)) {
        return false;
    }

    // TODO:
    // - Post or pre or any increment mode.
    DataType type_to_ldst = DataType::UNK;

    switch (data_type) {
    case 0:
        type_to_ldst = DataType::F32;
        break;

    case 1:
        type_to_ldst = DataType::INT16;
        break;

    case 2:
        type_to_ldst = DataType::INT8;
        break;

    default:
        break;
    }

    const int total_number_to_fetch = mask_count + 1;
    const int total_bytes_fo_fetch = get_data_type_size(type_to_ldst) * total_number_to_fetch;

    if (decoded_inst.opr.src1.bank == RegisterBank::IMMEDIATE) {
        decoded_inst.opr.src1.num *= get_data_type_size(type_to_ldst);
    }

    const bool is_store = decoded_inst.opcode == Opcode::STR;
    // right now proper repeat is implemented only for the store operation
    const int repeat_count = is_store ? mask_count : 0;
    set_repeat_multiplier(1, 1, 1, 1);
    BEGIN_REPEAT(repeat_count)
    GET_REPEAT(decoded_inst, RepeatMode::SLMSI)

    const int current_bytes_to_fetch = is_store ? 4 : total_bytes_fo_fetch;
    const int current_number_to_fetch = is_store ? 1 : total_number_to_fetch;

    const int to_store_offset = is_store ? src2_repeat_offset : 0;
    const int src0_offset = is_store ? src0_repeat_offset : 0;
    const int src1_offset = is_store ? dest_repeat_offset : 0;
    const int src2_offset = 0; // not used when storing

    // check if we handle this literal or texture read
    auto check_for_literal_texture_read = [&]() {
        if (mask_count > 0) {
            LOG_ERROR("Unimplemented literal buffer access with repeat");
            return true;
            return false;
        }
        if (decoded_inst.opr.dest.type != DataType::F32) {
            LOG_ERROR("Unimplemented non-f32 literal buffer access");
            return true;
            return false;
        }
        if (decoded_inst.opr.src1.bank != RegisterBank::IMMEDIATE || decoded_inst.opr.src2.bank != RegisterBank::IMMEDIATE) {
            LOG_ERROR("Unimplemented non-immediate literal buffer access");
            return true;
            return false;
        }
        if (is_store) {
            LOG_ERROR("Unhandled literal buffer store");
            return true;
            return false;
        }

        return true;
    };

    if (decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == program_state.texture_buffer_sa_offset) {
        // We are reading the texture buffer

        if (!check_for_literal_texture_read())
            return true;

        int offset = (program_state.texture_buffer_base + decoded_inst.opr.src1.num + decoded_inst.opr.src2.num) / sizeof(uint32_t);
        // we store the texture index in the first texture register, we don't do anything with the other 3
        if (offset % 4 != 0)
            continue;

        decoded_inst.opr.dest.type = DataType::INT32;
        variables.store(decoded_inst.opr.dest, fmt::format("{}", offset / 4), 0b1, 0);
        continue;
    } else if (decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == program_state.literal_buffer_sa_offset) {
        // We are reading the literal buffer

        if (!check_for_literal_texture_read())
            return true;

        int offset = program_state.literal_buffer_base + decoded_inst.opr.src1.num + decoded_inst.opr.src2.num;
        const uint8_t *literal_buffer = m_program.literal_buffer_data();
        const float literal = *reinterpret_cast<const float *>(literal_buffer + offset);

        variables.store(decoded_inst.opr.dest, fmt::format("{}", literal), 0b1, 0);
        continue;
    }

    std::string source_0 = variables.load(decoded_inst.opr.src0, 0b1, src0_offset);
    std::string source_1 = variables.load(decoded_inst.opr.src1, 0b1, src1_offset);

    // are we using the sa register containing the thread buffer address ?
    const bool is_thread_buffer_access = decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == program_state.thread_buffer_sa_offset;

    // Seems that if it's indexed by register, offset is in bytes and based on 0x10000?
    // Maybe that's just how the memory map operates. I'm not sure. However the literals on all shader so far is that
    // Another thing is that, when moe expand is not enable, there seems to be 4 bytes added before fetching... No absolute prove.
    // Maybe moe expand means it's not fetching after all? Dunno
    uint32_t REG_INDEX_BASE = is_thread_buffer_access ? 0x1000000 : 0x10000;

    if (decoded_inst.opr.src1.bank != shader::usse::RegisterBank::IMMEDIATE) {
        source_1 = fmt::format("{} - {}", source_1, REG_INDEX_BASE);
    }

    if (!moe_expand) {
        source_1 = fmt::format("{} + {}", source_1, 4);
    }

    if (!is_store) {
        std::string source_2 = variables.load(decoded_inst.opr.src2, 0b1, src2_offset);
        source_1 = fmt::format("{} + {}", source_1, source_2);
    }

    if (is_thread_buffer_access) {
        // We are reading the thread buffer

        // first some checks
        if (mask_count > 0) {
            LOG_ERROR("Unimplemented thread buffer access with repeat");
            return true;
        }
        if (decoded_inst.opr.dest.type != DataType::F32) {
            LOG_ERROR("Unimplemented non-f32 thread buffer access");
            return true;
        }

        if (program_state.thread_buffer_base != 0)
            source_1 = fmt::format("{} + {}", source_1, program_state.thread_buffer_base);

        std::string ptr = "thread_buffer[" + source_1  + " >> 2];";
        if (is_store) {
            std::string value = variables.load(decoded_inst.opr.dest, 0b1, 0);
            writer.add_to_current_body(ptr + " = " + value + ";");
        } else {
            variables.store(decoded_inst.opr.dest, ptr, 0b1, 0);
        }
        continue;
    }

    std::string base = fmt::format("{} + {}", source_0, source_1);

    if (m_features.support_memory_mapping) {
        LOG_ERROR("Memory mapping in not supported in GLSL");
        return true;
    } else  {
        if (is_store) {
            LOG_ERROR("Store opcode is not supported without memory mapping");
            return true;
        }

        // Address sometimes can be unaligned (start at mod 2 instead, as far as sunho wrote here)
        // If not we can just optimize it to load a vec4 at a time , haha
        writer.add_to_current_body(std::string("base_temp = ") + base + ";");

        for (int i = 0; i < total_bytes_fo_fetch / 4; ++i) {
            variables.store(decoded_inst.opr.dest, fmt::format("fetchMemory(base_temp + {})", i * 4), 0b1, 0);
            decoded_inst.opr.dest.num += 1;
        }
    }

    END_REPEAT()
    reset_repeat_multiplier();

    program_state.should_generate_vld_func = true;
    return true;
}

bool USSETranslatorVisitorGLSL::limm(
    bool skipinv,
    bool nosched,
    bool dest_bank_ext,
    bool end,
    Imm6 imm_value_bits26to31,
    ExtPredicate pred,
    Imm5 imm_value_bits21to25,
    Imm2 dest_bank,
    Imm7 dest_num,
    Imm21 imm_value_first_21bits) {
    decoded_inst.opcode = Opcode::MOV;

    std::uint32_t imm_value = imm_value_first_21bits | (imm_value_bits21to25 << 21) | (imm_value_bits26to31 << 26);

    decoded_inst.dest_mask = 0b1;
    decoded_inst.opr.dest = decode_dest(decoded_inst.opr.dest, dest_num, dest_bank, dest_bank_ext, false, 7, m_second_program);
    decoded_inst.opr.dest.type = DataType::UINT32;

    const std::string disasm_str = fmt::format("{:016x}: {}{}", m_instr, disasm::e_predicate_str(pred), disasm::opcode_str(decoded_inst.opcode));
    LOG_DISASM("{} {} #0x{:X}", disasm_str, disasm::operand_to_str(decoded_inst.opr.dest, 0b1, 0), imm_value);

    variables.store(decoded_inst.opr.dest, fmt::format("0x{:X}", imm_value), 0b1, 0);
    return true;
}
