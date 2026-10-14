# -*- coding: utf-8 -*-
"""Generate RPS Survival Market Final Report as Word document."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def set_run_font(run, name="Times New Roman", size=11, bold=False, italic=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(doc, text, *, size=11, bold=False, italic=False, align=None, space_after=6):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    return p


def add_heading_custom(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size={1: 16, 2: 14, 3: 12}.get(level, 11), bold=True)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return h


def add_formula(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, size=11, italic=True)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    return p


def set_cell_text(cell, text, *, bold=False, size=9, center=False):
    cell.text = ""
    p = cell.paragraphs[0]
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, size=9, center=True)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            set_cell_text(table.rows[r_idx + 1].cells[c_idx], val, size=9, center=True)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return table


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # Title
    add_para(doc, "RPS Survival Market", size=22, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_para(
        doc,
        "剪刀石头布生存交易战：完整项目报告",
        size=16,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=12,
    )
    add_para(doc, "课程：Quantitative Game Design / Group Project", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_para(doc, "组员：成员 A、B、C、D、E", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_para(doc, "日期：2026-09-28", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_para(doc, "版本：Final Report", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

    add_para(
        doc,
        "说明：本报告第 4–5 节使用一次 5 人模拟试玩数据，用于展示“日志表 → 数据分析 → 策略比较”的完整写法。"
        "正式提交时，可将模拟数据替换为课堂真实记录，分析框架与结论结构可直接沿用。",
        italic=True,
        size=10,
        space_after=12,
    )

    # Abstract
    add_heading_custom(doc, "摘要", 1)
    add_para(
        doc,
        "本报告设计并分析了一款 5 人课堂游戏 RPS Survival Market。游戏以剪刀石头布为不确定性对抗机制，"
        "结合星星、资金、卡牌交易与最终奇偶惩罚，模拟现实市场中的竞争、资源分配、风险决策与谈判。"
        "玩家在 2 轮内选择决斗或交易，最终通过手牌内部成对销毁、剩余卡牌扣星、比较最终星星与资金决定胜负。"
        "报告包括游戏规则、定量结构、预期策略、示例游戏数据、策略分析、局限与改进方向。",
    )
    add_para(doc, "核心判定链：", space_after=4)
    add_formula(
        doc,
        "个人手牌内部成对销毁 → 剩余卡牌扣星 → 比较最终星星 → 比较资金 → Sudden Death RPS",
    )

    # 1. Introduction
    add_heading_custom(doc, "1. Introduction", 1)
    add_heading_custom(doc, "1.1 游戏想法", 2)
    add_para(doc, "RPS Survival Market 的核心问题是：", space_after=4)
    add_para(
        doc,
        "在不确定对抗中，玩家应如何管理有限资源，使自己在最终清算时保留最多星星？",
        italic=True,
        space_after=8,
    )
    add_para(doc, "玩家同时管理三类资源：", space_after=4)
    for item in [
        "Cards 卡牌：既是决斗工具，也是最终惩罚来源。",
        "Stars 星星：代表生存能力，最终胜利的第一顺位。",
        "Money 资金：用于交易，也是同星时的第二顺位胜负标准。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(doc, "每轮玩家可选择：", space_after=4)
    for item in [
        "Duel 决斗：用 RPS 争夺星星，并改变双方手牌。",
        "Trade 交易：用卡牌、资金、星星进行谈判，调整手牌结构。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(
        doc,
        "最终阶段，玩家手牌内部相同卡牌成对销毁。每类卡牌剩余数量等于其数量 mod 2。"
        "每剩 1 张牌扣 1 星。最终星星最多者获胜；同星比资金；再同比 Sudden Death RPS。",
    )

    add_heading_custom(doc, "1.2 模拟的现实问题", 2)
    add_para(doc, "游戏模拟以下现实决策：", space_after=6)
    add_table(
        doc,
        ["游戏元素", "现实对应"],
        [
            ["卡牌", "资产或负债"],
            ["星星", "生存能力、市场地位"],
            ["资金", "财务资源"],
            ["决斗", "竞争性对抗、风险投资"],
            ["交易", "谈判、资源交换"],
            ["最终销毁", "冗余资产清算"],
            ["0 星危急", "破产边缘与紧急救助"],
            ["退出", "企业退出市场"],
            ["同星比资金", "同等生存能力下，财务实力决定胜负"],
        ],
    )
    add_para(
        doc,
        "因此，游戏不是纯运气或纯心理博弈，而是要求玩家用概率、期望值、奇偶性和资金管理做决策。",
    )

    add_heading_custom(doc, "1.3 项目目标", 2)
    add_para(doc, "本项目目标：", space_after=4)
    for i, item in enumerate(
        [
            "设计一个 20–30 分钟内可完成的课堂游戏。",
            "规则简单到 5 分钟能讲清。",
            "强制玩家记录决策与数字，形成可分析数据。",
            "比较不同策略：概率进攻、奇偶管理、卡牌干扰、资金保留。",
            "分析预期策略与实际行为差异。",
        ],
        1,
    ):
        add_para(doc, f"{i}. {item}", space_after=2)

    # 2. Game Design
    add_heading_custom(doc, "2. Game Design", 1)
    add_heading_custom(doc, "2.1 核心机制", 2)
    add_para(doc, "游戏流程：", space_after=4)
    add_formula(
        doc,
        "2 轮行动 → 最终卡牌销毁 → 剩余卡牌扣星 → 比较 Final Stars → 比较资金 → Sudden Death RPS。",
    )

    add_heading_custom(doc, "2.2 完整规则", 2)
    add_heading_custom(doc, "玩家人数", 3)
    add_para(
        doc,
        "每组 5 人，编号 Player 1–5。共进行 2 轮，行动顺序固定：1 → 2 → 3 → 4 → 5。",
    )

    add_heading_custom(doc, "初始资源", 3)
    add_table(
        doc,
        ["资源", "数量"],
        [
            ["Rock 石头", "4 张"],
            ["Paper 布", "4 张"],
            ["Scissors 剪刀", "4 张"],
            ["卡牌总数", "12 张"],
            ["Stars 星星", "3 颗"],
            ["Money 资金", "$60，建议 $10 × 6"],
        ],
    )

    add_heading_custom(doc, "信息规则", 3)
    add_para(doc, "隐藏信息：", space_after=2)
    add_para(doc, "• 每位玩家的具体手牌。", space_after=4)
    add_para(doc, "公开信息：", space_after=2)
    for item in [
        "当前星星与资金。",
        "公共弃牌区。",
        "决斗使用过的牌。",
        "全部交易内容。",
        "玩家是否退出、是否处于 0 星危急状态。",
        "每位玩家手牌总数。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "回合行动", 3)
    add_para(doc, "每位玩家在自己的回合只能选择一种主要行动：", space_after=2)
    add_para(doc, "• Action A：Duel 决斗", space_after=2)
    add_para(doc, "• Action B：Trade 交易", space_after=4)
    add_para(doc, "每轮限制：", space_after=2)
    add_para(doc, "• 每位玩家最多主动发起 1 次决斗。", space_after=2)
    add_para(doc, "• 每位玩家最多被挑战 1 次。", space_after=2)
    add_para(doc, "• 被挑战不消耗自己的主动行动。", space_after=6)

    add_heading_custom(doc, "决斗流程", 3)
    add_para(doc, "双方各从手牌中选 1 张，牌面朝下，同时公开。", space_after=4)
    add_para(doc, "RPS 规则：", space_after=2)
    add_para(doc, "• Rock 胜 Scissors", space_after=2)
    add_para(doc, "• Scissors 胜 Paper", space_after=2)
    add_para(doc, "• Paper 胜 Rock", space_after=4)
    add_para(doc, "相同卡牌：", space_after=2)
    add_para(doc, "• 两张牌立即销毁，放入公共弃牌区。", space_after=2)
    add_para(doc, "• 双方星星不变。", space_after=4)
    add_para(doc, "不同卡牌：", space_after=2)
    add_para(doc, "• 输家 Star −1，赢家 Star +1。", space_after=2)
    add_para(doc, "• 输家出的牌交给赢家，加入赢家手牌。", space_after=2)
    add_para(doc, "• 赢家出的牌仍保留在赢家手牌。", space_after=2)
    add_para(doc, "• 所有决斗用牌公开记录。", space_after=6)

    add_heading_custom(doc, "交易", 3)
    add_para(doc, "玩家选择 1 名其他仍在游戏中的玩家协商，限时 30 秒。", space_after=4)
    add_para(doc, "可交易：Cards 卡牌、Money 资金、Stars 星星。", space_after=4)
    add_para(doc, "限制：", space_after=2)
    for item in [
        "双方必须同时同意。",
        "正常交易不能让自己星星从 ≥1 降到 0。",
        "每笔正常交易最多转移 1 颗星星。",
        "交易内容公开。",
        "30 秒内未成交，行动结束。",
        "交易后，该回合不能主动决斗。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "0 星危急状态", 3)
    add_para(doc, "玩家因决斗输掉最后一颗星时，进入 Critical State：", space_after=2)
    for item in [
        "不能主动决斗。",
        "不能被挑战。",
        "可尝试通过交易获得星星。",
        "第一次变成 0 星时，获得一次紧急交易，限时 30 秒。",
        "若恢复至至少 1 星，继续游戏。",
        "若失败，必须 Withdraw 退出。",
        "退出后，资金和星星移出游戏，手牌全部放入公共弃牌区，不再重新分配。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "最终销毁", 3)
    add_para(
        doc,
        "2 轮结束后，每名玩家只能在自己的手牌内部，将两张相同类型的卡牌成对销毁。对每位玩家分别计算：",
        space_after=4,
    )
    add_formula(doc, "Remaining Rock = R mod 2")
    add_formula(doc, "Remaining Paper = P mod 2")
    add_formula(doc, "Remaining Scissors = S mod 2")
    add_formula(doc, "Remaining Cards = (R mod 2) + (P mod 2) + (S mod 2)")
    add_para(doc, "最终星星：", space_after=4)
    add_formula(doc, "Final Stars = max(0, Current Stars − Remaining Cards)")

    add_heading_custom(doc, "胜负条件", 3)
    add_para(doc, "按顺序判定：", space_after=2)
    add_para(doc, "1. Final Stars 最多者获胜。", space_after=2)
    add_para(doc, "2. 若同星，Remaining Money 多者获胜。", space_after=2)
    add_para(doc, "3. 若最终星星和资金都相同，Sudden Death RPS 一局定胜负。", space_after=6)

    add_heading_custom(doc, "2.3 设计原则", 2)
    add_para(doc, "游戏设计遵循以下原则：", space_after=4)
    for i, item in enumerate(
        [
            "规则可讲清：核心规则在 5 分钟内可解释完毕。",
            "时间可控：2 轮行动加最终清算，适合 22–27 分钟课堂。",
            "数学必要：概率、期望值、奇偶性和资金比较直接影响胜负。",
            "策略多样：决斗、交易、保留资金、调整手牌奇偶性均可形成不同策略。",
            "数据可记录：日志表记录概率估计、出牌、交易、星变、钱变和最终手牌。",
        ],
        1,
    ):
        add_para(doc, f"{i}. {item}", space_after=2)

    add_heading_custom(doc, "2.4 定量结构", 2)
    add_heading_custom(doc, "概率与期望值", 3)
    add_para(
        doc,
        "初始每种牌共 20 张。公共弃牌区、交易、决斗用牌公开，因此玩家可以追踪每种牌的剩余数量。"
        "决斗前估计对手出牌概率：P(Rock)、P(Paper)、P(Scissors)。只考虑星星收益时：",
        space_after=4,
    )
    add_formula(doc, "EV_Rock = P(Scissors) − P(Paper)")
    add_formula(doc, "EV_Paper = P(Rock) − P(Scissors)")
    add_formula(doc, "EV_Scissors = P(Paper) − P(Rock)")
    add_para(doc, "选择 EV 最大的牌；若 EV ≤ 0，理性玩家应倾向不主动决斗。")

    add_heading_custom(doc, "奇偶性", 3)
    add_para(doc, "最终每类剩余牌数：R mod 2、P mod 2、S mod 2。因此：", space_after=2)
    for item in [
        "偶数张同类牌最终可全部配对销毁，剩 0。",
        "奇数张同类牌最终剩 1，扣 1 星。",
        "初始每种 4 张，都是偶数，所以交易和决斗会改变奇偶性。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "资金与同分风险", 3)
    add_para(
        doc,
        "资金既用于交易，也决定同星时的胜负。花钱买牌时，要同时考虑：是否改善手牌奇偶性；"
        "是否避免最终扣星；资金减少是否导致同分落败。",
    )

    add_heading_custom(doc, "2.5 课堂流程与道具", 2)
    add_table(
        doc,
        ["阶段", "建议时间", "内容"],
        [
            ["Rule Explanation", "5 分钟", "发牌、讲规则、发日志表"],
            ["Round 1", "7 分钟", "5 人依次行动，交易限 30 秒"],
            ["Round 2", "7 分钟", "同上"],
            ["Final Cancellation", "3 分钟", "各自配对销毁、扣星、计分"],
            ["Result & Discussion", "5 分钟", "公布结果、对照日志、讨论概率与策略"],
            ["总计", "27 分钟", "可压缩到 22–25 分钟"],
        ],
    )
    add_para(doc, "道具：", space_after=2)
    for item in [
        "Rock / Paper / Scissors 卡牌：每种 20 张，共 60 张。",
        "星星标记：15 颗。",
        "资金代币：$10 × 30 张，共 $300。",
        "公共弃牌区标志 1 个。",
        "计时器 1 个。",
        "游戏日志表，每人 1 张。",
        "概率/EV 参考卡，每人 1 张。",
        "笔。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    # 3. Strategies
    add_heading_custom(doc, "3. Proposed Strategies", 1)
    add_heading_custom(doc, "Strategy 1：概率进攻", 2)
    add_para(
        doc,
        "根据公共弃牌区更新剩余牌比例。如果估计 P(Paper) 很高，则少出 Rock；如果 P(Scissors) 很高，则出 Rock。",
        space_after=4,
    )
    add_formula(doc, "EV_Rock = P(Scissors) − P(Paper)")
    add_formula(doc, "EV_Paper = P(Rock) − P(Scissors)")
    add_formula(doc, "EV_Scissors = P(Paper) − P(Rock)")

    add_heading_custom(doc, "Strategy 2：奇偶管理", 2)
    add_para(
        doc,
        "目标是让自己每类牌保持偶数。例如有 3 张 Paper，可以交易掉 1 张，或买进 1 张变成 4 张，避免最终剩 1 张扣星。",
    )

    add_heading_custom(doc, "Strategy 3：卡牌干扰", 2)
    add_para(
        doc,
        "赢走对手一张牌，可能让对手某类从 2 变 1，最终扣星。但自己也可能从 4 变 5，最终扣星。"
        "因此要计算双方奇偶变化。",
    )

    add_heading_custom(doc, "Strategy 4：信息策略", 2)
    add_para(doc, "记录公共弃牌区、交易、决斗用牌。每轮结束时更新：", space_after=2)
    for item in [
        "已销毁 Rock / Paper / Scissors 数量；",
        "已知在谁手中的牌；",
        "剩余未知牌比例。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "Strategy 5：资金保留", 2)
    add_para(
        doc,
        "资金决定同星胜负。买牌价格应低于“避免 1 星惩罚”的预期价值。"
        "例如花 $30 避免最终扣 1 星，可能值得；但如果会导致同分时输给对手，则未必。",
    )

    add_heading_custom(doc, "Strategy 6：0 星救助交易", 2)
    add_para(doc, "0 星玩家有紧急交易机会。其他玩家是否救他，取决于：", space_after=2)
    for item in [
        "获得资金或卡牌是否改善自己；",
        "对方继续存活是否会成为威胁；",
        "是否会影响最终同分排名。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    # 4. Gameplay Evidence
    add_heading_custom(doc, "4. Gameplay Evidence", 1)
    add_para(
        doc,
        "以下数据为一次 5 人模拟试玩数据，用于展示报告写法。",
        italic=True,
        space_after=8,
    )

    add_heading_custom(doc, "4.1 初始资源", 2)
    add_table(
        doc,
        ["玩家", "Rock", "Paper", "Scissors", "Stars", "Money"],
        [
            ["P1", "4", "4", "4", "3", "$60"],
            ["P2", "4", "4", "4", "3", "$60"],
            ["P3", "4", "4", "4", "3", "$60"],
            ["P4", "4", "4", "4", "3", "$60"],
            ["P5", "4", "4", "4", "3", "$60"],
        ],
    )

    add_heading_custom(doc, "4.2 行动日志摘要", 2)
    add_table(
        doc,
        ["轮次", "玩家", "行动", "对象", "我的出牌", "对方出牌", "结果", "星变", "钱变", "手牌变化"],
        [
            ["R1", "P1", "Duel", "P4", "Rock", "Scissors", "P1 赢", "P1 +1, P4 −1", "0", "P1 +1 S, P4 −1 S"],
            ["R1", "P2", "Trade", "P3", "—", "—", "成交", "0", "P2 −$20, P3 +$20", "P2 +1 R, P3 −1 R"],
            ["R1", "P3", "Duel", "P5", "Paper", "Rock", "P3 赢", "P3 +1, P5 −1", "0", "P3 +1 R, P5 −1 R"],
            ["R1", "P4", "Duel", "P2", "Scissors", "Paper", "P4 赢", "P4 +1, P2 −1", "0", "P4 +1 P, P2 −1 P"],
            ["R1", "P5", "Duel", "P1", "Paper", "Rock", "P5 赢", "P5 +1, P1 −1", "0", "P5 +1 R, P1 −1 R"],
            ["R2", "P1", "Duel", "P2", "Scissors", "Paper", "P1 赢", "P1 +1, P2 −1", "0", "P1 +1 P, P2 −1 P"],
            ["R2", "P2", "Duel", "P5", "Rock", "Scissors", "P2 赢", "P2 +1, P5 −1", "0", "P2 +1 S, P5 −1 S"],
            ["R2", "P3", "Trade", "P5", "—", "—", "成交", "0", "P3 −$20, P5 +$20", "P3 +1 R, P5 −1 R"],
            ["R2", "P4", "Duel", "P3", "Paper", "Rock", "P4 赢", "P4 +1, P3 −1", "0", "P4 +1 R, P3 −1 R"],
            ["R2", "P5", "Duel", "P1", "Scissors", "Paper", "P5 赢", "P5 +1, P1 −1", "0", "P5 +1 P, P1 −1 P"],
        ],
    )

    add_heading_custom(doc, "4.3 最终计分", 2)
    add_table(
        doc,
        ["玩家", "当前星星", "资金", "最终手牌 R/P/S", "剩余卡数", "Final Stars", "排名"],
        [
            ["P1", "3", "$60", "3/4/5", "2", "1", "第二/第三加赛"],
            ["P2", "2", "$40", "5/2/5", "2", "0", "第五"],
            ["P3", "3", "$60", "4/4/4", "0", "3", "第一"],
            ["P4", "4", "$60", "5/5/3", "3", "1", "第二/第三加赛"],
            ["P5", "3", "$80", "3/5/3", "3", "0", "第四"],
        ],
    )
    add_para(doc, "计算示例：", space_after=4)
    add_para(
        doc,
        "• P1：Rock 3 → 1，Paper 4 → 0，Scissors 5 → 1，剩余 2 张。Final Stars = max(0, 3 − 2) = 1",
        space_after=2,
    )
    add_para(
        doc,
        "• P3：Rock 4 → 0，Paper 4 → 0，Scissors 4 → 0，剩余 0 张。Final Stars = max(0, 3 − 0) = 3",
        space_after=2,
    )
    add_para(
        doc,
        "• P4：Rock 5 → 1，Paper 5 → 1，Scissors 3 → 1，剩余 3 张。Final Stars = max(0, 4 − 3) = 1",
        space_after=6,
    )

    add_heading_custom(doc, "4.4 概率估计与 EV 示例", 2)
    add_table(
        doc,
        ["决斗", "玩家", "估计 P(R)/P(P)/P(S)", "出牌", "计算 EV", "实际结果"],
        [
            ["R1 P1 vs P4", "P1", "0.33 / 0.33 / 0.34", "Rock", "0.34 − 0.33 = +0.01", "赢"],
            ["R1 P3 vs P5", "P3", "0.40 / 0.30 / 0.30", "Paper", "0.40 − 0.30 = +0.10", "赢"],
            ["R2 P1 vs P2", "P1", "0.30 / 0.40 / 0.30", "Scissors", "0.40 − 0.30 = +0.10", "赢"],
            ["R2 P4 vs P3", "P4", "0.35 / 0.30 / 0.35", "Paper", "0.35 − 0.35 = 0.00", "赢"],
            ["R2 P5 vs P1", "P5", "0.30 / 0.40 / 0.30", "Scissors", "0.40 − 0.30 = +0.10", "赢"],
        ],
    )

    # 5. Analysis
    add_heading_custom(doc, "5. Analysis", 1)
    add_heading_custom(doc, "5.1 预期策略 vs 实际行为", 2)
    add_para(doc, "示例数据中，玩家总体表现：", space_after=2)
    for item in [
        "共 10 次主动行动，其中 8 次决斗，2 次交易。",
        "没有出现相同卡牌决斗，因此公共弃牌区无销毁牌。",
        "交易集中在 $20 换 1 张牌。",
        "最终 P3 获胜，P1 与 P4 同星同钱进入 Sudden Death。",
        "P5 虽然最终星为 0，但资金 $80 高于 P2 的 $40，因此排在 P2 前。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(doc, "观察：", space_after=4)
    add_para(
        doc,
        "1. 玩家更倾向决斗，而不是交易。原因可能是决斗直接改变星星，反馈快；交易需要谈判，30 秒内难达成复杂协议。",
        space_after=4,
    )
    add_para(
        doc,
        "2. 交易确实改变奇偶性。P3 用 $20 买 1 Rock，使自己的 Rock 从 3 变 4，避免最终 Rock 剩余 1。"
        "但 P5 卖出 1 Rock 后，Rock 从 4 变 3，最终剩余 1，被扣 1 星。"
        "这说明交易是双刃剑：改善自己可能损害对方。",
        space_after=4,
    )
    add_para(
        doc,
        "3. 资金 tie-breaker 很重要。P5 最终星 0，但资金 $80，因此排在 P2 之前。"
        "如果 P5 没有保留资金，可能排名更低。",
    )

    add_heading_custom(doc, "5.2 EV 分析", 2)
    add_para(
        doc,
        "示例中，P1 在 R1 对 P4 的 EV 为 +0.01，几乎中性；R2 对 P2 的 EV 为 +0.10，较有利。两次都赢。"
        "但样本只有 8 次决斗，不能证明 EV 高一定赢。EV 分析的价值在于：",
        space_after=2,
    )
    for item in [
        "帮助玩家在决斗前理性选择出牌；",
        "避免只凭心理猜测；",
        "让“是否主动决斗”有数字依据。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(doc, "如果 EV ≤ 0，主动决斗的期望星星收益为负，玩家应倾向交易或保存优势。")

    add_heading_custom(doc, "5.3 奇偶性分析", 2)
    add_para(doc, "最终剩余卡数：", space_after=6)
    add_table(
        doc,
        ["玩家", "R mod 2", "P mod 2", "S mod 2", "剩余卡数", "扣星"],
        [
            ["P1", "1", "0", "1", "2", "−2"],
            ["P2", "1", "0", "1", "2", "−2"],
            ["P3", "0", "0", "0", "0", "0"],
            ["P4", "1", "1", "1", "3", "−3"],
            ["P5", "1", "1", "1", "3", "−3"],
        ],
    )
    add_para(
        doc,
        "P3 是典型成功案例：最终三类牌都是偶数，剩余 0 张，不扣星，因此以 3 星获胜。"
        "P4 是典型失败案例：虽然当前星星最高，为 4 星，但最终 Rock、Paper、Scissors 都是奇数，"
        "剩余 3 张，扣 3 星，最终只剩 1 星。这说明：赢牌不一定有利，卡牌数量多不一定好。",
    )

    add_heading_custom(doc, "5.4 资金与排名分析", 2)
    add_table(
        doc,
        ["玩家", "Final Stars", "Money", "排名"],
        [
            ["P3", "3", "$60", "1"],
            ["P1", "1", "$60", "第二/第三加赛"],
            ["P4", "1", "$60", "第二/第三加赛"],
            ["P5", "0", "$80", "4"],
            ["P2", "0", "$40", "5"],
        ],
    )
    add_para(
        doc,
        "P5 证明了资金保留策略的价值：虽然最终星星为 0，但同星时资金最多，因此排名高于 P2。"
        "P1 和 P4 同星同钱，只能 Sudden Death RPS，说明资金管理直接影响是否要多打一局。",
    )

    add_heading_custom(doc, "5.5 交易分析", 2)
    add_para(doc, "两笔交易均为 $20 换 1 张牌。平均交易价格：$20 / Card。", space_after=4)
    add_para(doc, "交易效果：", space_after=2)
    for item in [
        "P2 买 Rock：R 从 4 变 5，奇数化，长期不利。",
        "P3 买 Rock：R 从 3 变 4，偶数化，有利。",
        "P5 卖 Rock：R 从 4 变 3，奇数化，不利。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(
        doc,
        "因此，同一笔交易对双方影响不同。理性玩家在交易前应计算："
        "Δ Remaining Cards = 交易后剩余卡数 − 交易前剩余卡数。"
        "如果交易使自己剩余卡数减少，则值得；如果增加，则应要求更高补偿。",
    )

    add_heading_custom(doc, "5.6 模型局限", 2)
    for i, item in enumerate(
        [
            "样本小：只有 5 人、2 轮、8 次决斗，统计显著性低。",
            "概率估计主观：玩家估计的 P(R)/P(P)/P(S) 未必准确。",
            "交易时间短：30 秒可能限制复杂谈判。",
            "无相同卡牌决斗：示例中公共弃牌区为空，未体现销毁机制。",
            "模拟数据：本报告第 4–5 节为示例，实际结论需用真实数据验证。",
        ],
        1,
    ):
        add_para(doc, f"{i}. {item}", space_after=2)

    # 6. Conclusion
    add_heading_custom(doc, "6. Conclusion", 1)
    add_heading_custom(doc, "6.1 对现实问题的模拟效果", 2)
    add_para(doc, "RPS Survival Market 较好地模拟了以下现实问题：", space_after=2)
    for item in [
        "不确定性下的竞争决策；",
        "有限资源在生存、财务、资产之间的分配；",
        "谈判与交易的双刃剑效应；",
        "冗余资产在清算时变成负担；",
        "资金在同等生存能力下的 tie-breaker 作用；",
        "破产边缘的紧急救助。",
    ]:
        add_para(doc, "• " + item, space_after=2)
    add_para(doc, "游戏的核心数学结构清晰：", space_after=4)
    add_formula(doc, "概率 → EV → 决斗决策")
    add_formula(doc, "卡牌数量 mod 2 → 最终扣星")
    add_formula(doc, "Final Stars → Money → Sudden Death RPS")

    add_heading_custom(doc, "6.2 局限", 2)
    for item in [
        "2 轮可能不足以让策略充分展开。",
        "交易限时 30 秒可能抑制复杂谈判。",
        "取消退出后手牌重分配，减少了资源再流通。",
        "玩家可能更关注心理博弈，而非概率计算。",
        "日志表需要裁判或玩家认真填写，否则数据质量低。",
    ]:
        add_para(doc, "• " + item, space_after=2)

    add_heading_custom(doc, "6.3 可能的改进", 2)
    for i, item in enumerate(
        [
            "若时间允许，恢复 3 轮，但每轮行动时间压缩到 5 分钟。",
            "增加“相同卡牌决斗”触发概率，例如允许玩家用 2 张牌组合决斗。",
            "引入拍卖式交易：每轮公开一张牌，玩家用资金竞拍。",
            "使用预设概率机器人，与玩家决斗，检验玩家 EV 判断。",
            "设计电子日志表，自动计算 mod 2、Final Stars 和 EV。",
            "增加“市场事件卡”，随机改变某种牌的销毁规则。",
        ],
        1,
    ):
        add_para(doc, f"{i}. {item}", space_after=2)

    # 7. Individual Contribution
    add_heading_custom(doc, "7. Individual Contribution", 1)
    add_table(
        doc,
        ["成员", "主要负责内容", "具体贡献"],
        [
            ["A", "规则设计", "规则精简、时间控制、决斗与交易流程"],
            ["B", "数学建模", "EV 公式、奇偶性、Final Stars 计算、示例数据"],
            ["C", "道具与日志表", "卡牌、星星、资金代币、日志表设计"],
            ["D", "课堂主持与测试", "组织试玩、计时、收集日志、整理数据"],
            ["E", "报告撰写与展示", "报告整合、PPT、结果讨论、最终校对"],
        ],
    )

    # Appendix A
    add_heading_custom(doc, "Appendix A：游戏日志表模板", 1)
    add_heading_custom(doc, "A1. 行动记录表", 2)
    add_table(
        doc,
        [
            "玩家",
            "轮次",
            "行动类型",
            "对象",
            "赛前估计",
            "我的出牌",
            "对方出牌",
            "结果",
            "星变",
            "钱变",
            "手牌变化",
            "公共弃牌",
            "备注",
        ],
        [["", "", "", "", "", "", "", "", "", "", "", "", ""], ["", "", "", "", "", "", "", "", "", "", "", "", ""]],
    )

    add_heading_custom(doc, "A2. 最终计分表", 2)
    add_table(
        doc,
        ["玩家", "当前星星", "资金", "最终手牌 R/P/S", "剩余卡数", "Final Stars", "排名"],
        [["", "", "", "", "", "", ""], ["", "", "", "", "", "", ""]],
    )

    add_heading_custom(doc, "A3. 公共弃牌区统计", 2)
    add_table(
        doc,
        ["轮次", "Rock 销毁", "Paper 销毁", "Scissors 销毁", "备注"],
        [["R1", "", "", "", ""], ["R2", "", "", "", ""]],
    )

    # Appendix B
    add_heading_custom(doc, "Appendix B：课堂流程卡", 1)
    add_table(
        doc,
        ["阶段", "时间", "主持人指令"],
        [
            ["规则讲解", "5 min", "发初始资源，讲 RPS、决斗、交易、最终销毁"],
            ["Round 1", "7 min", "P1→P5 依次行动，交易限 30 秒"],
            ["Round 2", "7 min", "同上，更新公共弃牌区"],
            ["Final Cancellation", "3 min", "各自配对销毁，计算 Final Stars"],
            ["结果与讨论", "5 min", "公布排名，比较 EV、奇偶、资金策略"],
            ["总计", "27 min", "可压缩至 22–25 分钟"],
        ],
    )

    # Appendix C
    add_heading_custom(doc, "Appendix C：概率 / EV 快速参考卡", 1)
    add_table(
        doc,
        ["出牌", "胜", "负", "EV 公式"],
        [
            ["Rock", "Scissors", "Paper", "EV_Rock = P(S) − P(P)"],
            ["Paper", "Rock", "Scissors", "EV_Paper = P(R) − P(S)"],
            ["Scissors", "Paper", "Rock", "EV_Scissors = P(P) − P(R)"],
        ],
    )
    add_para(doc, "最终剩余卡数：", space_after=4)
    add_formula(doc, "Remaining Cards = (R mod 2) + (P mod 2) + (S mod 2)")
    add_para(doc, "最终星星：", space_after=4)
    add_formula(doc, "Final Stars = max(0, Current Stars − Remaining Cards)")

    # Appendix D
    add_heading_custom(doc, "Appendix D：一句话总结", 1)
    add_para(
        doc,
        "RPS Survival Market 用最简单的 RPS 对抗，承载了概率、期望值、奇偶性、资源交易和资金 tie-breaker "
        "等定量决策，适合在 25 分钟左右完成一轮完整的“设计—试玩—记录—分析”课堂项目。",
    )

    out = Path(__file__).resolve().parent / "RPS-Survival-Market-Final-Report.docx"
    doc.save(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    build()
