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

#include <shader/spirv/translator.h>

#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/SpvBuilder.h>

#include <gxm/types.h>
#include <shader/decoder_helpers.h>
#include <shader/disasm.h>
#include <shader/types.h>
#include <util/log.h>

#include <numeric>

using namespace shader;
using namespace usse;

bool USSETranslatorVisitorSpirv::vmov(
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

    // TODO: adjust dest mask if needed
    CompareMethod compare_method = CompareMethod::NE_ZERO;
    spv::Op compare_op = spv::OpAny;

    const DataType test_type = is_u8_conditional ? DataType::UINT8 : move_data_type;
    const bool is_test_signed = is_signed_integer_data_type(test_type);
    const bool is_test_unsigned = is_unsigned_integer_data_type(test_type);
    const bool is_test_integer = is_test_signed || is_test_unsigned;

    if (is_conditional) {
        compare_method = static_cast<CompareMethod>((test_bit_2 << 1) | test_bit_1);

        switch (compare_method) {
        case CompareMethod::LT_ZERO:
            if (is_test_unsigned)
                compare_op = spv::Op::OpULessThan;
            else if (is_test_signed)
                compare_op = spv::Op::OpSLessThan;
            else
                compare_op = spv::Op::OpFOrdLessThan;
            break;
        case CompareMethod::LTE_ZERO:
            if (is_test_unsigned)
                compare_op = spv::Op::OpULessThanEqual;
            else if (is_test_signed)
                compare_op = spv::Op::OpSLessThanEqual;
            else
                compare_op = spv::Op::OpFOrdLessThanEqual;
            break;
        case CompareMethod::NE_ZERO:
            if (is_test_integer)
                compare_op = spv::Op::OpINotEqual;
            else
                compare_op = spv::Op::OpFOrdNotEqual;
            break;
        case CompareMethod::EQ_ZERO:
            if (is_test_integer)
                compare_op = spv::Op::OpIEqual;
            else
                compare_op = spv::Op::OpFOrdEqual;
            break;
        }
    }

    // Recompile

    m_b.setLine(m_recompiler.cur_pc);

    if ((move_data_type == DataType::F16) || (move_data_type == DataType::F32)) {
        set_repeat_multiplier(2, 2, 2, 2);
    } else {
        set_repeat_multiplier(1, 1, 1, 1);
    }

    BEGIN_REPEAT(repeat_count)
    GET_REPEAT(decoded_inst, RepeatMode::SLMSI);

    spv::Id source_to_compare_with_0 = spv::NoResult;
    spv::Id source_1 = load(decoded_inst.opr.src1, decoded_dest_mask, src1_repeat_offset);
    spv::Id source_2 = spv::NoResult;
    spv::Id result = spv::NoResult;

    if (source_1 == spv::NoResult) {
        LOG_ERROR("Source not Loaded");
        return false;
    }

    if (is_conditional) {
        source_to_compare_with_0 = load(decoded_inst.opr.src0, decoded_dest_mask, src0_repeat_offset);
        source_2 = load(decoded_inst.opr.src2, decoded_dest_mask, src2_repeat_offset);
        spv::Id result_type = m_b.getTypeId(source_2);
        spv::Id v0_comp_type = is_test_unsigned ? m_b.makeUintType(32) : (is_test_signed ? m_b.makeIntType(32) : m_b.makeFloatType(32));
        spv::Id v0_type = utils::make_vector_or_scalar_type(m_b, v0_comp_type, m_b.getNumComponents(source_2));
        spv::Id v0 = utils::make_uniform_vector_from_type(m_b, v0_type, 0);

        bool source_2_first = false;

        if (compare_op != spv::OpAny) {
            // Merely do what the instruction does
            // First compare source0 with vector 0
            spv::Id cond_result = m_b.createOp(compare_op, utils::make_vector_or_scalar_type(m_b, m_b.makeBoolType(), m_b.getNumComponents(source_to_compare_with_0)),
                { source_to_compare_with_0, v0 });

            // For each component, if the compare result is true, move the equivalent component from source1 to dest,
            // else the same thing with source2
            // This behavior matches with OpSelect, so use it. Since IMix doesn't exist (really)
            result = m_b.createOp(spv::OpSelect, result_type, { cond_result, source_1, source_2 });
        } else {
            // We optimize the float case. We can make the GPU use native float instructions without touching bool or integers
            // Taking advantage of the mix function: if we use absolute 0 and 1 as the lerp, we got the equivalent of:
            // mix(a, b, c) with c.comp is either 0 or 1 <=> if c.comp == 0 return a else return b.
            switch (compare_method) {
            case CompareMethod::LT_ZERO: {
                // For each component: if source0.comp < 0 return 0 else return 1
                // That means if we use mix, it should be mix(src1, src2, step_result)
                result = m_b.createBuiltinCall(result_type, std_builtins, GLSLstd450Step, { v0, source_to_compare_with_0 });
                source_2_first = false;
                break;
            }

            case CompareMethod::LTE_ZERO: {
                // For each component: if 0 < source0.comp return 0 else return 1
                // Or, if we turn it around: if source0.comp <= 0 return 1 else return 0
                // That means if we use mix, it should be mix(src2, src1, step_result)
                result = m_b.createBuiltinCall(result_type, std_builtins, GLSLstd450Step, { source_to_compare_with_0, v0 });
                source_2_first = true;
                break;
            }

            case CompareMethod::NE_ZERO:
            case CompareMethod::EQ_ZERO: {
                // Taking advantage of the sign and absolute instruction
                // The sign instruction returns 0 if the component equals to 0, else 1 if positive, -1 if negative
                // That means if we absolute the sign result, we got 0 if component equals to 0, else we got 1.
                // src2 will be first for Not equal case.
                result = m_b.createBuiltinCall(result_type, std_builtins, GLSLstd450FSign, { source_to_compare_with_0 });
                result = m_b.createBuiltinCall(result_type, std_builtins, GLSLstd450FAbs, { result });

                if (compare_method == CompareMethod::NE_ZERO) {
                    source_2_first = true;
                }

                break;
            }

            default: {
                LOG_ERROR("Unknown compare method: {}", static_cast<int>(compare_method));
                return false;
            }
            }

            // Mixing!! I'm like a little witch!!
            result = m_b.createBuiltinCall(result_type, std_builtins, GLSLstd450FMix, { source_2_first ? source_2 : source_1, source_2_first ? source_1 : source_2, result });
        }
    } else {
        result = source_1;
    }

    store(decoded_inst.opr.dest, result, decoded_dest_mask, dest_repeat_offset);

    END_REPEAT()

    reset_repeat_multiplier();
    return true;
}

bool USSETranslatorVisitorSpirv::vpck(
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

    // Recompile
    m_b.setLine(m_recompiler.cur_pc);

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

    spv::Id source = load(decoded_inst.opr.src1, dest_mask, src1_repeat_offset);

    if (source == spv::NoResult) {
        LOG_ERROR("Source not loaded");
        return false;
    }

    if (decoded_inst.opr.src2.type != DataType::UNK) {
        Operand src1 = decoded_inst.opr.src1;
        Operand src2 = decoded_inst.opr.src2;
        src1.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        src2.swizzle = SWIZZLE_CHANNEL_4_DEFAULT;
        spv::Id source1 = load(src1, 0b11, src1_repeat_offset);
        spv::Id source2 = load(src2, 0b11, src2_repeat_offset);
        source = utils::finalize(m_b, source1, source2, decoded_inst.opr.src1.swizzle, m_b.makeIntConstant(0), dest_mask);
    }

    // source is float destination is int
    if (is_float_data_type(decoded_inst.opr.dest.type) && !is_float_data_type(decoded_inst.opr.src1.type)) {
        source = utils::convert_to_float(m_b, m_util_funcs, source, decoded_inst.opr.src1.type, scale);
    }

    // source is int destination is float
    if (!is_float_data_type(decoded_inst.opr.dest.type) && is_float_data_type(decoded_inst.opr.src1.type)) {
        source = utils::convert_to_int(m_b, m_util_funcs, source, decoded_inst.opr.dest.type, scale);
    }

    store(decoded_inst.opr.dest, source, dest_mask, dest_repeat_offset);
    END_REPEAT()

    reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitorSpirv::vldst(
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

    if (decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == m_spirv_params.texture_buffer_sa_offset) {
        // We are reading the texture buffer

        if (!check_for_literal_texture_read())
            return true;

        int offset = (m_spirv_params.texture_buffer_base + decoded_inst.opr.src1.num + decoded_inst.opr.src2.num) / sizeof(uint32_t);
        // we store the texture index in the first texture register, we don't do anything with the other 3
        if (offset % 4 != 0)
            continue;

        decoded_inst.opr.dest.type = DataType::INT32;
        store(decoded_inst.opr.dest, m_b.makeIntConstant(offset / 4), 0b1);
        continue;
    } else if (decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == m_spirv_params.literal_buffer_sa_offset) {
        // We are reading the literal buffer

        if (!check_for_literal_texture_read())
            return true;

        int offset = m_spirv_params.literal_buffer_base + decoded_inst.opr.src1.num + decoded_inst.opr.src2.num;
        const uint8_t *literal_buffer = m_program.literal_buffer_data();
        const float literal = *reinterpret_cast<const float *>(literal_buffer + offset);

        store(decoded_inst.opr.dest, m_b.makeFloatConstant(literal), 0b1);
        continue;
    }

    spv::Id source_0 = load(decoded_inst.opr.src0, 0b1, src0_offset);
    spv::Id source_1 = load(decoded_inst.opr.src1, 0b1, src1_offset);

    // are we using the sa register containing the thread buffer address ?
    const bool is_thread_buffer_access = decoded_inst.opr.src0.bank == RegisterBank::SECATTR && decoded_inst.opr.src0.num == m_spirv_params.thread_buffer_sa_offset;

    // Seems that if it's indexed by register, offset is in bytes and based on 0x10000?
    // Maybe that's just how the memory map operates. I'm not sure. However the literals on all shader so far is that
    // Another thing is that, when moe expand is not enable, there seems to be 4 bytes added before fetching... No absolute prove.
    // Maybe moe expand means it's not fetching after all? Dunno
    uint32_t REG_INDEX_BASE = is_thread_buffer_access ? 0x1000000 : 0x10000;
    spv::Id reg_index_base_cst = m_b.makeIntConstant(REG_INDEX_BASE);
    spv::Id i32_type = m_b.makeIntType(32);

    if (decoded_inst.opr.src1.bank != shader::usse::RegisterBank::IMMEDIATE) {
        source_1 = m_b.createBinOp(spv::OpISub, m_b.getTypeId(source_1), source_1, reg_index_base_cst);
    }

    if (!moe_expand) {
        source_1 = m_b.createBinOp(spv::OpIAdd, i32_type, source_1, m_b.makeIntConstant(4));
    }

    if (!is_store) {
        spv::Id source_2 = load(decoded_inst.opr.src2, 0b1, src2_offset);
        source_1 = m_b.createBinOp(spv::OpIAdd, i32_type, source_1, source_2);
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

        if (m_spirv_params.thread_buffer_base != 0)
            source_1 = m_b.createBinOp(spv::OpIAdd, i32_type, source_1, m_spirv_params.thread_buffer_base);

        // get the index in the float array
        spv::Id index = m_b.createBinOp(spv::OpShiftRightLogical, i32_type, source_1, m_b.makeUintConstant(2));
        spv::Id float_ptr = utils::create_access_chain(m_b, spv::StorageClassPrivate, m_spirv_params.thread_buffer, { index });
        if (is_store) {
            spv::Id value = load(decoded_inst.opr.dest, 0b1);
            m_b.createStore(value, float_ptr);
        } else {
            spv::Id value = m_b.createLoad(float_ptr, spv::NoPrecision);
            store(decoded_inst.opr.dest, value, 0b1);
        }
        continue;
    }

    spv::Id base = m_b.createBinOp(spv::OpIAdd, i32_type, source_0, source_1);

    if (m_features.support_memory_mapping) {
        utils::buffer_address_access(m_b, m_spirv_params, m_util_funcs, m_features, decoded_inst.opr.dest, to_store_offset, base, get_data_type_size(type_to_ldst), current_number_to_fetch, -1, is_store);
    } else {
        if (is_store) {
            LOG_ERROR("Store opcode is not supported without memory mapping");
            return true;
        }

        for (int i = 0; i < total_bytes_fo_fetch / 4; ++i) {
            spv::Id offset = m_b.createBinOp(spv::OpIAdd, m_b.makeIntType(32), base, m_b.makeIntConstant(4 * i));
            spv::Id src = utils::fetch_memory(m_b, m_spirv_params, m_util_funcs, offset);
            store(decoded_inst.opr.dest, src, 0b1);
            decoded_inst.opr.dest.num += 1;
        }
    }

    END_REPEAT()
    reset_repeat_multiplier();

    return true;
}

bool USSETranslatorVisitorSpirv::limm(
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
    Instruction inst;
    inst.opcode = Opcode::MOV;

    std::uint32_t imm_value = imm_value_first_21bits | (imm_value_bits21to25 << 21) | (imm_value_bits26to31 << 26);
    spv::Id const_imm_id = m_b.makeUintConstant(imm_value);

    inst.dest_mask = 0b1;
    inst.opr.dest = decode_dest(inst.opr.dest, dest_num, dest_bank, dest_bank_ext, false, 7, m_second_program);
    inst.opr.dest.type = DataType::UINT32;

    const std::string disasm_str = fmt::format("{:016x}: {}{}", m_instr, disasm::e_predicate_str(pred), disasm::opcode_str(inst.opcode));
    LOG_DISASM("{} {} #0x{:X}", disasm_str, disasm::operand_to_str(inst.opr.dest, 0b1, 0), imm_value);

    store(inst.opr.dest, const_imm_id, 0b1);
    return true;
}
