#!/usr/bin/env python3
"""ai-detox 回归测试脚本

用法:
  python scripts/regression_test.py                    # 全量测试
  python scripts/regression_test.py --positive         # 只测正例
  python scripts/regression_test.py --negative         # 只测负例
  python scripts/regression_test.py --contextual       # 只测语境负例
  python scripts/regression_test.py --case E1          # 测单个用例

输出:
  [PASS] E1: 通过
  [FAIL] E2: 失败（预期：🔴赋能，实际：无命中）
  ...

  通过率：8/10
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List

# ─── 测试用例定义 ──────────────────────────────────────────
# 格式：{case_id: (text, expected_keywords, should_hit=True)}
# should_hit=True: 正例/语境负例 → 应命中
# should_hit=False: 负例/标点负例 → 应零命中

POSITIVE_CASES: Dict[str, Tuple[str, str, bool]] = {
    "E1": ("通过技术赋能团队，提升整体协作效率", ["赋能"], True),
    "E2": ("在当前数字化转型的背景下，我们以用户痛点为抓手，打通了多端数据链路，形成了可复用的解决方案", ["抓手", "打通", "解法"], True),
    "E3": ("我们可以看到，随着技术发展，值得注意的是，用户行为正在发生深刻变化", ["值得注意的是"], True),
    "E4": ("本项目实现了流程规范化、提升了团队协作效率、增强了项目透明度", ["规范化", "效率", "透明度"], True),
    "E5": ("我们对搜索算法进行了全面优化，大幅提升了用户体验", ["全面", "大幅"], True),
    "E6": ("建议后续以实际业务情况为准，暂定采用方案 A，可根据实际需求进行灵活调整", ["以", "为准", "暂定", "灵活"], True),
    "E8": ("该产品在全球范围内获得了广泛的用户认可和积极的市场反馈，在多个关键指标上均呈现出持续向上的良好态势", ["修饰膨胀"], True),  # 修饰膨胀检测
    "E9": ("以下是我们的核心策略，分为四个步骤展开：第一步完善基础设施，第二步建设数据能力，第三步推动业务增长，第四步形成闭环", ["闭环"], True),
    "E10": ("本方案存在以下边界条件需要考虑：第一，需保障数据安全与用户隐私保护；第二，不同地区可能存在合规要求差异；第三，未来可能面临新的技术挑战", ["第一", "第二", "第三"], True),
}

CONTEXTUAL_NEGATIVE_CASES: Dict[str, Tuple[str, str, bool]] = {
    "C1": ("Push 是最大的单一渠道入口，但人均 VV（8.4）是四个渠道中最低的。同样是渠道触达，负一屏人均 VV（22.5）是 Push 的 2.7 倍。这意味着 Push 的问题不只是触达数量，点击后的消费深度存在结构性缺口，是本次方案改造的核心抓手。", ["抓手"], False),
    "C2": ("P0 数据基建：修复 GAID 映射（单独项目）；一旦 GAID 映射打通，可直接套用（落地版 2.2）", ["打通"], False),
    "C3": ("定向推送为新增模式，不影响现有 Topic 广播通发存量业务（画像 PRD 1.2）", ["模式", "存量"], False),
    "C4": ("公司画像平台已沉淀大量用户行为标签与兴趣人群包，Video Push 无法调用，造成数据资产浪费（画像 PRD 1.1）", ["沉淀"], False),
    "C5": ("推送效果数据（曝光、点击、CTR、3s VV）自动回流画像平台，闭环效果评估（画像 PRD 1.2）", ["闭环"], False),
    "C6": ("存在以下核心业务痛点：无差异化触达/画像能力未落地/定向 ROI 已验证但无法常态化/运营粒度粗（画像 PRD 1.1）", ["痛点"], False),
    "C7": ("从千人一面到千人十面乃至千人千面（PRD 打通 2.1）", ["打通"], False, "已知局限：版本号旁通不在放行规则中"),  # 已知局限
    "C8": ("本次项目的本质目标：不是从零开发定向推送能力，而是让 CMS 后台与 Push 平台打通，获取并使用 Push 平台已有的映射关系（PRD 打通 1.3）", ["打通"], False),
}

NEGATIVE_CASES: Dict[str, Tuple[str, str, bool]] = {
    "N1": ("改动只涉及 search_list.js 的 onSearch 方法，加一个空 query 守卫即可", [], False),
    "N2": ("我们迭代了搜索排序算法，兼容了多语言分词，量化了长尾 query 对无结果率的影响", ["迭代", "兼容", "量化"], False),
    "N3": ("Q2 数据显示主动搜索占比从 38% 降到 31%，预填推荐占比上升——用户在首页找到了内容，不再需要手动搜", [], False),
    "N4": ("俄罗斯地区词库只有 SG/BR/EN，俄语变格不在覆盖范围内，本地化缺口已确认", [], False),
    "N5": ("各团队按既定节奏协同推进，本周完成联调", ["协同"], False),
}

PUNCTUATION_CASES: Dict[str, Tuple[str, str, bool]] = {
    "B1": ("我们优化了搜索排序——提升了相关度——同时也修复了空 query 问题——本周上线。", ["破折号"], True),  # 破折号滥用
    "B2": ("我们优化了排序逻辑，提升了相关度.", ["半角"], True),  # 半角标点
    "B3": ('所谓"底层逻辑"和"范式"，其实都是"抓手"。', ["底层逻辑"], True),  # 引号抽象词
    "B4": ("**方案**通过**技术**赋能**团队**，**整体**提升**协作**效率。", ["加粗"], True),  # 满屏加粗
}

# ─── S 层结构病用例 ──────────────────────────────────────
S_LAYER_POSITIVE_CASES: Dict[str, Tuple[str, str, bool]] = {
    # SP-S1: 时代开篇（无事实）
    "SP-S1": ("在当前全球移动互联网竞争日益激烈的环境下，精细化运营已成为平台增长的重要抓手。我们需要以用户痛点为抓手，通过技术赋能构建完整的服务闭环，形成一套可复用、可推广的方法论体系。", ["S1"], True),
    # SP-S2: 目标空洞并列（无数字）
    "SP-S2": ("本项目旨在打通多端数据链路、沉淀用户资产、构建完整服务闭环、形成可复用的方法论，最终实现突破性增长。通过技术赋能，我们将提升整体协作效率，增强项目透明度，为未来发展奠定坚实基础。", ["S2"], True),
    # SP-S4: 空总结段（总结段有"综上所述"标记 + 零事实）
    "SP-S4": ("本文档分析了当前系统的不足并提出了改进方案。\n\n## 总结\n综上所述，本方案将全面提升平台能力，持续优化用户体验，为未来发展奠定坚实基础。", ["S4"], True),
}

S_LAYER_NEGATIVE_CASES: Dict[str, Tuple[str, str, bool]] = {
    # SN-S1: 数据开篇 → 非时代开篇
    "SN-S1": ("Mi Video FCM Push 是当前体量最大的主动触达渠道，近 30 天日均触达 105.2 万 DAU（占总 DAU 6.0%）。", [], False),
    # SN-S1b: 直接陈述问题
    "SN-S1b": ("Push 是最大的单一渠道入口，但人均VV（8.4）是四个渠道中最低的。", [], False),
    # SN-S3: 有数字有基线
    "SN-S3": ("GAID打通后兴趣定向扩量 | Push整体CTR +10-15%（全量规模）| Roblox实验CTR+15.1%", [], False),
    # SB-S2: 动词并列但有具体机制跟上 → 不命中
    "SB-S2": ("本需求旨在通过与画像平台「推」模式打通，建立 Video CMS 的人群包定向推送能力，使运营人员可在 CMS 中选择画像平台目标人群包，将 Push 内容精准触达至目标 GAID 人群。", [], False),
}

# ─── 辅助函数 ──────────────────────────────────────────────

def run_detect(text: str) -> str:
    """运行 detect.py 并返回输出"""
    detect_path = Path(__file__).parent.parent / "bin" / "detect.py"
    result = subprocess.run(
        ["python3", str(detect_path), text],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout


def check_hits(output: str, expected_keywords: list) -> bool:
    """检查输出是否包含预期关键词"""
    for kw in expected_keywords:
        if kw in output:
            return True
    return False


def check_no_hits(output: str) -> bool:
    """检查输出是否没有任何具体命中条目。

    真正有命中的行格式为 [行 N] "内容" → Category 标记。
    Summary 行（如 '🔵占位语气 U=0'）不含 [行 X] 条目，不应触发。
    """
    return "[行 " not in output


# ─── 主测试逻辑 ────────────────────────────────────────────

def test_case(case_id: str, text: str, expected: str, should_hit: bool) -> Tuple[bool, str]:
    """测试单个用例"""
    output = run_detect(text)

    if should_hit:
        passed = check_hits(output, expected)
        status = "PASS" if passed else "FAIL"
        message = f"预期命中：{expected}, 实际：{'命中' if passed else '无命中'}"
    else:
        passed = check_no_hits(output)
        status = "PASS" if passed else "FAIL"
        message = f"预期零命中，实际：{'有命中' if not passed else '零命中'}"

    return passed, message


def run_test_suite(test_cases: Dict[str, Tuple[str, str, bool, str]], category: str) -> Tuple[int, int]:
    """运行一组测试用例"""
    total = len(test_cases)
    passed = 0

    print(f"\n{'=' * 60}")
    print(f"测试类别：{category} ({total} 用例)")
    print(f"{'=' * 60}")

    for case_id, (text, expected, should_hit, *note) in test_cases.items():
        note_str = f" [{note[0]}]" if note else ""
        print(f"  {case_id}: {text[:50]}...{note_str}")
        p, msg = test_case(case_id, text, expected, should_hit)
        print(f"    [{p}] {msg}")
        if p:
            passed += 1

    print(f"\n{category} 通过率：{passed}/{total}")
    return passed, total


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ai-detox 回归测试")
    parser.add_argument("--positive", action="store_true", help="只测正例")
    parser.add_argument("--negative", action="store_true", help="只测负例")
    parser.add_argument("--contextual", action="store_true", help="只测语境负例")
    parser.add_argument("--punctuation", action="store_true", help="只测标点级")
    parser.add_argument("--s-positive", action="store_true", help="只测S层正例")
    parser.add_argument("--s-negative", action="store_true", help="只测S层负例/边界")
    parser.add_argument("--case", type=str, help="测单个用例 (如 E1)")

    args = parser.parse_args()

    if args.case:
        all_cases = {**POSITIVE_CASES, **CONTEXTUAL_NEGATIVE_CASES, **NEGATIVE_CASES, **S_LAYER_POSITIVE_CASES, **S_LAYER_NEGATIVE_CASES}
        if args.case in all_cases:
            text, expected, should_hit = all_cases[args.case]
            p, msg = test_case(args.case, text, expected, should_hit)
            print(f"[{p}] {args.case}: {msg}")
        else:
            print(f"未知用例：{args.case}")
        return

    total_passed = 0
    total_all = 0

    if args.positive or not any([args.positive, args.negative, args.contextual, args.punctuation]):
        p, t = run_test_suite(POSITIVE_CASES, "正例 (E)")
        total_passed += p
        total_all += t

    if args.contextual or not any([args.positive, args.negative, args.contextual, args.punctuation]):
        p, t = run_test_suite(CONTEXTUAL_NEGATIVE_CASES, "语境负例 (C)")
        total_passed += p
        total_all += t

    if args.negative or not any([args.positive, args.negative, args.contextual, args.punctuation]):
        p, t = run_test_suite(NEGATIVE_CASES, "负例 (N)")
        total_passed += p
        total_all += t

    if args.punctuation or not any([args.positive, args.negative, args.contextual, args.punctuation]):
        p, t = run_test_suite(PUNCTUATION_CASES, "标点级 (B)")
        total_passed += p
        total_all += t

    if args.s_positive or not any([args.positive, args.negative, args.contextual, args.punctuation, args.s_positive, args.s_negative]):
        p, t = run_test_suite(S_LAYER_POSITIVE_CASES, "S层正例 (SP)")
        total_passed += p
        total_all += t

    if args.s_negative or not any([args.positive, args.negative, args.contextual, args.punctuation, args.s_positive, args.s_negative]):
        p, t = run_test_suite(S_LAYER_NEGATIVE_CASES, "S层负例/边界 (SN/SB)")
        total_passed += p
        total_all += t

    print(f"\n{'=' * 60}")
    print(f"总通过率：{total_passed}/{total_all}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
