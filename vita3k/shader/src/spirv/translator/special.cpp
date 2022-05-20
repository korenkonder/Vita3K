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

#include <SPIRV/SpvBuilder.h>

#include <shader/decoder_helpers.h>
#include <shader/disasm.h>
#include <shader/types.h>
#include <util/log.h>

using namespace shader;
using namespace usse;

bool USSETranslatorVisitorSpirv::smbo(Imm1 nosched,
    Imm12 dest_offset,
    Imm12 src0_offset,
    Imm12 src1_offset,
    Imm12 src2_offset) {
    LOG_DISASM("{:016x}: SMBO {}, {}, {}, {}", m_instr, dest_offset, src0_offset, src1_offset, src2_offset);

    m_b.setLine(m_recompiler.cur_pc);

    auto parse_offset = [&](const int idx, Imm12 offset) {
        for (int i = 0; i < 17; i++) {
            repeat_increase[idx][i] = i + offset;
        }
    };
    parse_offset(3, dest_offset);
    parse_offset(0, src0_offset);
    parse_offset(1, src1_offset);
    parse_offset(2, src2_offset);

    return true;
}

bool USSETranslatorVisitorSpirv::kill(
    ShortPredicate pred) {
    LOG_DISASM("{:016x}: KILL {}", m_instr, disasm::s_predicate_str(pred));

    m_b.setLine(m_recompiler.cur_pc);
    m_b.makeStatementTerminator(spv::OpKill, "kill");

    return true;
}

bool USSETranslatorVisitorSpirv::depthf(
    bool sync,
    bool src0_bank_ext,
    bool end,
    Imm1 src1_bank_ext,
    Imm1 src2_bank_ext,
    bool nosched,
    ShortPredicate pred,
    bool two_sided,
    Imm2 feedback,
    Imm1 src0_bank,
    Imm2 src1_bank,
    Imm2 src2_bank,
    Imm7 dest_n,
    Imm7 src0_n,
    Imm7 src1_n,
    Imm7 src2_n) {
    if (!USSETranslatorVisitor::depthf(sync, src0_bank_ext, end, src1_bank_ext, src2_bank_ext, nosched, pred, two_sided,
        feedback, src0_bank, src1_bank, src2_bank, dest_n, src0_n, src1_n, src2_n)) {
        return false;
    }

    m_b.setLine(m_recompiler.cur_pc);

    if (frag_depth_id == 0) {
        frag_depth_id = m_b.createVariable(spv::NoPrecision, spv::StorageClassOutput, type_f32, "gl_FragDepth");
        m_b.addDecoration(frag_depth_id, spv::DecorationBuiltIn, spv::BuiltInFragDepth);
    }

    spv::Id depth = load(decoded_inst.opr.src0, 0b1);
    m_b.createStore(depth, frag_depth_id);

    return true;
}
