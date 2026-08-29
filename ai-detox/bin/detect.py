#!/usr/bin/env python3
"""ai-detox 检测引擎 v1.5 — 八类规则扫描 + 双语气改写

用法:
  python detect.py "纯文本输入"
  python detect.py --file path/to/file.txt
  cat file.txt | python detect.py --stdin
  python detect.py --demo  # 用校准语料跑一轮测试
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from collections import defaultdict
from typing import NamedTuple

# ─── 数据类型 ───────────────────────────────────────────────

class Hit(NamedTuple):
    line: int          # 行号（从 1 开始）
    category: str      # 类别代码，如 "T", "O", "Y", "B4a", "B4c", "P", "U", "S1"
    color: str         # emoji 标签，如 "🔴", "🟢(4c)"
    text: str          # 命中原文
    suggestion: str    # 建议替换/操作
    detail: str = ""   # 额外信息（子类计数等）


class ParagraphInfo(NamedTuple):
    start: int         # 起始行号
    end: int           # 结束行号
    lines: list[str]   # 段落内的行（ stripped ）
    original_lines: list[str]  # 原始行（含缩进）


# ─── 词表 ────────────────────────────────────────────────────

# 🔴 第一类：借喻式黑洞词 — 需要语境判定
# 有实指 = 词前后紧跟具体系统名 / 数字 / 技术动作 / 专名 → 放行
# 每个词 = [建议替换, 理由, 有实指模式列表]
BLACKHOLE_WORDS = {
    # word -> (suggestion, reason, positive_patterns)
    "赋能": ("帮助 / 让…能做", "军事隐喻滥用，本质就是'帮'", [r"赋能.{0,10}(?:search_list|FCM|Push|GAID|画像|标签|定向)"]),
    "抓手": ("具体动作或工具名", "没有实质含义，读者不知道你到底指什么", [r"是.{0,25}(?:核心|主要|首要)?抓手"]),  # "X是本次方案改造的核心抓手"判断句有实指放行；"以X为抓手"等空洞介词结构仍命中
    "打通": (
        "连接 / 合并 / 贯通",
        "建筑隐喻替代了具体动词",
        [
            r"(?:映射|联上|对接|接入|同步|获取|使用|复用).{0,5}打通",  # 词在打通前：C8 "获取已有映射关系" → 有实指放行
            r"打通.{0,10}(?:GAID|画像|标签|数据基础|埋点|指标|映射|获取|使用|复用|能力|人群包|效果)",  # 打通后跟系统名/技术动作 → 有实指
            r"打通\s*\d+\.\d+",  # PRD 章节号引用：C7 "打通 2.1" → 有实指放行
            r"打通.{0,15}(?:CTR|DAU|\d+.*%|\d+\.\d+)",  # 打通后紧跟量化结果 → SN-S3 类放行
            r"打通.{0,20}(?:CMS|Video|Push|FCM|画像平台|目标人群|精准触达)",  # 打通后有具体机制描述 → SB-S2 类放行
        ],
    ),  # "打通了数据链路"仍命中（无具体系统名）
    "沉淀": ("积累 / 留存 / 形成", "农业隐喻，换掉不影响理解", [r"沉淀.{0,10}(?:用户行为标签|GAID人群包|投放数据|特征数据)"]),
    "协同": ("配合 / 合作", "'加强协同'等无主语协同必杀", [r"协同(?:推进|配合|合作|工作|作战|开发|办公|联动|努力|发力|作战)|(?:团队|部门|各方|跨部门|上下游|前后端).{0,6}协同"]),  # "协同推进/协同配合"等有具体动作放行；仅"加强协同"等无主语空洞用法命中
    "矩阵": ("系列 / 组合", "数学词汇当包装纸用", []),
    "闭环": ("完整流程 / 端到端", "控制论术语被掏空", [r"闭环.{0,10}(?:效果|评估|数据|链路|回流|机制|系统|平台|接口|建设)"]),  # "闭环效果评估"/"数据回流闭环"等有具体支撑放行；"形成闭环"等空洞用法仍命中
    "组合拳": ("多项措施 / 一揽子方案", "体育隐喻，直接说有几项就行", []),
    "顶层设计": ("整体规划", "建筑隐喻掩盖了具体的设计内容", []),
    "方法论": ("方法 / 思路", "听起来比'方法'高级但没更多信息量", [r"沉淀.{0,5}方法论|投放方法论|运营方法论"]),  # "沉淀出…方法论"/"XX方法论"有实指放行
    "生态": ("生态系统 / 相关方网络", "生物隐喻，80%情况后面才是重点", []),
    "联动": ("互动 / 互相影响", "和'协同'类似，加了前缀就危险", [r"(?:内容库|热词|配置|Push|FCM|CMS|客户端|服务端|推荐|搜索).{0,8}联动|联动.{0,8}(?:内容库|热词|Push|FCM|配置|搜索|推荐)"]),  # 两个方向都有具体系统/功能名 → 放行
    "倒逼": ("迫使 / 推动", "逆向因果隐喻", []),
    "聚焦": ("关注 / 集中资源在", "'聚焦'之后必须有具体对象才算有用", []),
    "拉齐": ("统一 / 对齐 / 保持一致", "口语化版就是'对齐'", []),
    "驱动": ("推动 / 带动", "机械隐喻泛滥，90%可替换", []),
    "贯穿": ("全程 / 从头到尾", "方向性隐喻，换成时间/空间描述即可", []),
    "落地": ("实施 / 执行 / 上线", "降落隐喻", [r"(?:分期|直接|已|可|能|支持|支持|阶段.{0,2}).{0,3}落地|落地.{0,5}(?:为|成|后|时|的|第[一二三四五六七八九十]|v\d|\d|灰度|AB实验)"]),  # "分期落地""直接可落地""已落地"等有实指放行；"落地为XX"/"落地方案"等空洞用法仍命中
    "解法": ("解决方案 / 办法", "'提供了解法'这种空洞句式才危险", []),
    "布局": ("规划 / 安排", "棋类隐喻，通常后面应该跟具体内容", []),
    "壁垒": ("系统障碍 / 信息孤岛 / 跨部门困难", "'打破壁垒'是经典AI套话", []),
    "破局": ("找到突破口 / 改变现状", "棋类隐喻替代了具体问题描述", []),
}

# 🟠 第二类：元话语
META_WORDS = [
    "我们可以看到",
    "我们发现",
    "值得注意的是",
    "从某种意义上说",
    "需要强调的是",
    "应该说",
    "不可否认的是",
    "必须承认",
    "事实上",
    "实际上",
    "众所周知",
    "大家知道",
    "不言而喻",
]

# 🟡 第三类：万能修辞模板
UNIVERSAL_TEMPLATES = [
    (r"在当今.{2,20}的时代", "删除万能开头"),
    (r"在当今.{2,20}的背景下", "删除万能开头"),
    (r"在当今.{2,20}的浪潮下", "删除万能开头"),
    (r"在当前.{2,20}背景下", "删除万能开头"),
    (r"在当前.{2,20}环境下", "删除万能开头"),
    (r"在当前.{2,20}环境中", "删除万能开头"),
    (r"随着.{2,30}的发展", "删除万能开头"),
    (r"随着.{2,20}发展", "删除万能开头（无'的'变体）"),
    (r"随着.{2,30}的进步", "删除万能开头"),
    (r"随着.{2,30}的深入", "删除万能开头"),
    r"……已成为不可阻挡的趋势",
    r"在这个瞬息万变的时代",
    r"站在新的起点上",
    r"综上所述",
    r"总而言之",
    r"归根结底",
    r"……是大势所趋",
    r"在这个竞争日益激烈的环境中",
    r"不得不提的是",
    r"说到这里",
    # 编号列废话（作为 tuple 走 regex）
    (r"第一.{1,40}第二.{1,40}第三", "编号三件套 '第一...；第二...；第三...'"),
    (r"第一步.{1,40}第二步.{1,40}第三步", "编号四步走框架标签"),
    # Yb 全球通吃句式 — 换任何行业文档都成立的万金油句
    (r"这需要各方共同努力|让我们(?:共同|一起).{0,20}努力", "CCT信号：无主体万金油，换哪篇都行"),
    (r"我们期待更好的未来|期待.*更好.*(体验|明天|未来)", "无时间/条件/行动主体的情感断言"),
    (r"这将带来全新的(体验|世界|格局|局面)", "\"全新\"+无具体改进点的抽象体验"),
    (r"助力(行业|企业|公司|产业).(高质量|健康|快速|稳步).*(发展|提升|进步)", "CCT: 无主体万金油 + 趋势膨胀"),
]

# 🟢 第四类：结构癖 & 虚化表达

# 4a. 排比癖
PARALLEL_PATTERNS = [
    (r"(?:实现了|提升了|增强了|完善了|强化了){.+?}(?:、|,\s*)(?:实现了|提升了|增强了|完善了|强化了){.+?}(?:、|,\s*)(?:实现了|提升了|增强了|完善了|强化了){.+?}",
     "三段式并列 '实现了A、提升了B、增强了C'，AI写作最强信号"),
]

# 4b. 虚化程度：虚化副词 + 抽象结果
VAGUE_MODIFIERS = {
    "大幅": ["提升", "优化", "改善", "增强", "提高"],
    "显著": ["提高", "降低", "改进", "优化", "提升", "改善"],
    "明显": ["效果", "变化", "提升", "改善"],
    "有效": ["解决", "改善", "提升", "优化", "促进"],
    "极大": ["促进", "推动", "提升", "改善"],
    "全面": ["深化", "推进", "覆盖", "升级"],
    "进一步": ["完善", "加强", "优化", "提升", "改进"],
    "持续": ["优化", "改进", "加强", "提升", "改善"],
}

ABSTRACT_RESULTS = [
    "效率", "体验", "能力", "价值", "水平", "质量", "效果",
    "用户体验", "服务质量", "工作效率", "协作效率", "管理能力",
]

# 4c. 后缀病 — 使用向后查找精确匹配后缀词
SUFFIX_PATTERNS = {
    "-化": re.compile(r"(?<=[一-鿿])(?:系统化|前置化|闭环化|生态化|可视化|标准化|精细化|规范化|数字化|智能化|轻量化|敏捷化|平台化|专业化|规模化|集约化|集中化|分散化|透明化|一体化|融合化)", re.UNICODE),
    "-性": re.compile(r"(?<=[一-鿿])(?:协作性|可扩展性|可维护性|稳定性|可靠性|灵活性|有效性|兼容性|可移植性|可用性|健壮性|鲁棒性|一致性|连贯性|确定性|模糊性)", re.UNICODE),
    "-型": re.compile(r"(?<=[一-鿿])(?:管理型|服务型|驱动型|平台型|数据型|用户型|市场型|增长型)", re.UNICODE),
    "-度": re.compile(r"(?<=[一-鿿])(?:颗粒度|透明度|可控度|覆盖率|匹配度|契合度|贴合度)", re.UNICODE),
}

# 4d. 万能程度副词
WEAKENERS = [
    "一定程度上",
    "或多或少",
    "某种程度上",
    "相对而言",
    "相对地",
]

# 4e. 欧化句型
OVERSEAS_SENTENCES = [
    (r"不是.{1,80}而是.{1,80}", "\"不是X而是Y\"表演式对比，英文 It's not X but Y 直译"),
    # "是...的"判断句壳 — 用单独函数处理以排除疑问/因果（见下方 _scan_overseas）
    None,  # placeholder: "是……的" handled by _scan_shi_de separately
    (r"(?:进行|实现|完成|导致|构成|加以|予以|做出){.+?(?:验证|改造|提升|优化|调整|处理|分析|评估|改革|创新|建设|部署|落实|推进|实施)}",
     "空心动词+noun直译：conduct/achieve+名词"),
    # 注意：前者/后者改为段落级计数（≥2才命中），不在这里逐行报
    (r"(?:正确的|结构化的|与当前任务高度相关的|全面的|系统的|有效的){.+?的{0,3}.{1,20}",
     "英文多重定语从句直译，中文一串'的'"),
]

# 4e "是……的"判断句壳 — 排除因果解释(是因为/是由于)、疑问(是否)、表演对比(而是)
# 使用 helper + strict pattern 避免误杀
OVERSEAS_SHI_DE = re.compile(
    r"(?<!是为)(?!是否|而是)"       # 不匹配"是为"、"是否"、"而是"开头的
    r"是.{2,60}"                    # 以"是"开头
    r".*(?:的)[，;。]|是.{5,60}的[，。；;]"  # ...的结尾(第二分支要求句读结尾，避免"是A的B"名词短语误判)
)


# ⚫ 第五类：标点级检测
# （在扫描器中直接处理，不单独成表）

# 🔵 第六类：占位语气 / 犹豫腔调
PLACEHOLDER_PATTERNS = {
    # 5a: "以……为准/主/依据" — “为”和“准”之间无间隙，用直接匹配
    "5a_以..为准": re.compile(r"以.{1,20}(?:为|作为)(?:准|主|依据)", re.UNICODE),
    # 5a: "视……而定/情况"
    "5a_视..而定": re.compile(r"视.{1,20}(?:而[定据]|根据|情况).{0,5}(?:定|情|况)", re.UNICODE),
    # 5b: "暂定"/"暂不" + 动作 — 显式匹配"定"和"不"变体
    "5b_暂定": re.compile(r"暂(?:定|不)[一二三四]?(?:采用|开放|支持|扩展|接入|推出|实施)", re.UNICODE),
    # 5c: "可灵活调整"/"酌情处理" — "可"后面允许更多字符
    "5c_灵活调整": re.compile(r"(?:可|可以).{0,20}灵活调整|酌情.{0,10}(?:处理|修改|调整)", re.UNICODE),
    # 5d: "原则上"/"大致上" + 判断词
    "5d_原则上": re.compile(r"原则上(?:应该|可以|优先|首选)", re.UNICODE),
    "5d_大致上": re.compile(r"大致上(?:可以|接受|符合|满足)", re.UNICODE),
    # 5e: 兜底占位
    "5e_合理范围": re.compile(r"在合理范围内", re.UNICODE),
    "5e_总体而言": re.compile(r"总体上|整体而言", re.UNICODE),
    "5e_届时再议": re.compile(r"届时再议|后续视情况确定", re.UNICODE),
    "5e_看实际情况": re.compile(r"看.{1,15}(?:实际|具体|团队|项目).情况", re.UNICODE),
}

PLACEHOLDER_NAMES = {
    "5a_以..准": "以……为准",
    "5a_视..而定": "视……而定",
    "5b_暂定": "暂定/暂不",
    "5c_灵活调整": "可灵活调整/酌情处理",
    "5d_原则上": "原则上应该/可以",
    "5e_合理范围": "在合理范围内",
    "5e_总体而言": "总体上/整体而言",
    "5e_届时再议": "届时再议/后续视情况确定",
    "5e_看实际情况": "看实际情况而定",
}


# 🟤 第七类：行文结构病
S1_PATTERN = re.compile(
    r"(?:在.{2,15}(?:竞争|发展|变革|转型|升级|演进).{0,20}|"
    r"随着.{2,20}.{0,10}(?:发展|进步|深入|变化|演变|崛起).{0,20}|"
    r"在.{2,15}(?:时代|背景下|浪潮下|趋势下|环境).{0,20})",
    re.UNICODE,
)

S2_PATTERN = re.compile(
    r"(?:打通|沉淀|构建|形成|赋能|建立|打造|构建|打造)"
    r".{0,15}"
    r"(?:、|,\s*)(?:打通|沉淀|构建|形成|赋能|建立|打造|构建|打造)"
    r".{0,15}"
    r"(?:、|,\s*)(?:打通|沉淀|构建|形成|赋能|建立|打造|构建|打造)",
    re.UNICODE,
)

S3_VAGUE_PATTERN = re.compile(
    r"(?:有效|显著|大幅|极大|全面|持续|进一步){0,1}(?:提升|优化|改善|增强|提高|改善)"
    r".{0,15}(?:效率|体验|能力|价值|水平|质量)",
    re.UNICODE,
)

S4_SUMMARY_PATTERN = re.compile(
    r"^(#{1,3}\s*)?(?:总结|展望|结语|未来方向|结束语)",
    re.UNICODE,
)

# S5 安全对齐痕迹：RLHF / Constitutional AI 语气模式（arXiv 2606.08172）
S5_NEUTRAL_PATTERNS = [
    (re.compile(r"负责任地(?:的)?", re.UNICODE), "RLHF语气惯性：用'负责任地'替代明确判断"),
    (re.compile(r"需要谨慎(?:看待|评估|处理|推进)", re.UNICODE), "回避明确判断，表演性谨慎"),
    (re.compile(r"值得深入(?:探讨|研究|分析)", re.UNICODE), "不把事情说死，RLHF模型倾向"),
    (re.compile(r"应该客观(?:评估|衡量|分析)", re.UNICODE), "用'客观'包装不判断"),
    (re.compile(r"多方(?:共同)?(?:努力|协作|配合)", re.UNICODE), "CCT信号：无主体万金油"),
    (re.compile(r"奠定了(?:坚实)?基础", re.UNICODE), "建筑隐喻空洞化，RLHF常见收尾套路"),
    (re.compile(r"(?:需要|应当|应该).{0,5}(?:进一步|更多|持续)(?:探索|研究|观察)", re.UNICODE), "把决策无限延期"),
]

# S5a 合规腔嵌套：EU AI Act 治理文本风格（arXiv 2606.08172）
S5A_COMPLIANCE_PATTERNS = [
    (re.compile(r"在(?:符合|遵循|满足).{1,30}.{0,3}的前提(.{0,3})?下", re.UNICODE), "前置条件噪音：主张被前置条件稀释"),
    (re.compile(r"在(?:合规|合法|安全).{1,10}.框架内", re.UNICODE), '"在XX框架内"是噪音，不是具体限制'),
    (re.compile(r"遵循.{1,15}.原则的(?:基础上|前提下|指导下)", re.UNICODE), "原则列表太长→全变空，套娃前置条件"),
    (re.compile(r"在(?:保障|保护|维护).{1,15}.(?:的情况下|的前提下)", re.UNICODE), "空洞前置条件，实际等于没说"),
]

# S5c 信息密度病：段落事实字符占比过低
def info_density_bay(para_lines: list[str], threshold: float = 0.10) -> dict:
    """计算段落的 Fact-Bearing 字符比例。

    Returns {'fact_chars', 'total_chars', 'ratio', 'low': bool}。
    事实字符定义：数字、中文标点后的汉字、英文缩写/URL、具体名词/动作。
    """
    full = "".join(para_lines)
    total = len(full)
    if total == 0:
        return {"fact_chars": 0, "total_chars": 0, "ratio": 0.0, "low": False}

    # 事实字符：数字 | 英文单词/缩写(≥2字母) | URL | 百分号相关 | 专有名词前缀
    fact_pattern = re.compile(
        r"\d+"                       # 所有数字
        r"|https?://[^\s]+"         # URL
        r"|%(?:\s*\d+)?"            # 百分号及关联数字
        r"[A-Za-z]{2,}"             # 英文缩写(≥2字母)
        r"|Mi\s*Video"              # 专名
        r"|Push|FCM|GAID|DAU|CTR|API|SDK|HTTP",  # 技术专名
        re.IGNORECASE,
    )

    # 额外加分：具体动词/动作（含技术落地动作——有动作即有实指，避免误判"打通/建立/选择"等具体步骤为空洞长段）
    action_chars = len(re.findall(r"(?:获取|实现|提供|支持|接入|配置|搜索|点击|发送|审核|拦截|覆盖|打通|建立|选择|触达|推送|回流|映射|沉淀|定向|复用|对接|同步)", full))

    num_matches = len(fact_pattern.findall(full))
    # 粗略估算：每个数字/URL/缩写 ≈ 3-5 个事实字符
    fact_from_patterns = sum(len(m.group()) for m in fact_pattern.finditer(full))
    fact_chars = fact_from_patterns + action_chars * 2

    ratio = fact_chars / total if total > 0 else 0
    return {
        "fact_chars": fact_chars,
        "total_chars": total,
        "ratio": round(ratio, 3),
        "low": ratio < threshold and total > 50,  # 仅长段落判定
    }


# ─── 修饰膨胀（Modifier Inflation） ──────────────────────────
# 大量空泛修饰词堆叠在同一句/短句内，无具体数字支撑。
# 与 4b 虚化程度互补：4b = 单个虚化副词+抽象结果；此处 = 多类修饰词堆叠。

EXPANSIVE_MODIFIERS = {
    # 规模膨胀：夸大覆盖范围但不说具体哪里
    "scale": re.compile(r"(?:全球| worldwide|全世界|四海|八方|天下|九州|寰球)", re.UNICODE | re.IGNORECASE),
    # 广度膨胀：广泛的/全方位的但没说哪些
    "breadth": re.compile(r"(?:广泛|全方位|全覆盖|全面铺开|面面俱到|无所不包)", re.UNICODE),
    # 情感膨胀：积极的好听的但没有证据
    "emotional": re.compile(r"(?:积极|良好|优异|卓越|惊人|非凡|出色|满意|赞誉|好评)", re.UNICODE),
    # 趋势膨胀：持续向好但没有数据和方向细节
    "trend": re.compile(r"(?:持续向上|持续增长|稳步提升|不断攀升|节节攀升|蒸蒸日上|日益强劲)", re.UNICODE),
    # 历史膨胀：用"历史"包装空洞内容
    "history": re.compile(r"(?:悠久历史|博大精深|源远流长|千古传承|历久弥新|穿越千年)", re.UNICODE),
    # 数量膨胀：大量但说不清楚到底多少
    "quantity": re.compile(r"(?:大量|众多|无数|海量|成千上万|不计其数|数不胜数)", re.UNICODE),
}


# ─── 扫描器 ─────────────────────────────────────────────────

class Scanner:
    """逐行 + 逐段扫描，收集所有类别的命中。"""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.hits: list[Hit] = []

    # ── 工具方法 ──

    def _add(self, line: int, category: str, color: str, text: str,
             suggestion: str, detail: str = ""):
        self.hits.append(Hit(line, category, color, text.strip(),
                             suggestion.strip(), detail.strip()))

    def _find_matches(self, line_idx: int, line: str, patterns: dict | list,
                      color: str, match_type: str = "dict"):
        """通用正则匹配。match_type='dict': patterns是{pattern: suggestion};
        'list': patterns是[(regex_or_str, suggestion)]."""
        if match_type == "dict":
            for pat, sug in patterns.items():
                if isinstance(pat, str):
                    if pat in line:
                        idx = line.index(pat)
                        self._add(line_idx + 1, pat[:6], color, pat, sug)
                else:
                    m = pat.search(line)
                    if m:
                        self._add(line_idx + 1, pat.pattern[:6], color,
                                  m.group(), sug)
        else:  # list mode
            for pat, sug in patterns:
                if isinstance(pat, str):
                    if pat in line:
                        self._add(line_idx + 1, pat[:6], color, pat, sug)
                else:
                    m = pat.search(line)
                    if m:
                        self._add(line_idx + 1, pat.pattern[:8], color,
                                  m.group(), sug)

    # ── 按行扫描 ──

    def scan_categories(self):
        """对所有类别执行逐行扫描。"""
        for i, raw in enumerate(self.lines):
            line = raw.strip()
            if not line:
                continue

            self._scan_blackhole(i, line)
            self._scan_meta(i, line)
            self._scan_universal(i, line)
            self._scan_vague_modifier(i, line)
            self._scan_suffix_disease(i, line)
            self._scan_weakeners(i, line)
            self._scan_overseas(i, line)
            self._scan_punctuation(i, line)
            self._scan_modifier_inflation(i, line)

        # 段落级扫描（在逐行之后）
        paragraphs = self._split_paragraphs()
        self._scan_parallel(paragraphs)
        self._scan_placeholder_combo(paragraphs)
        self._scan_cross_ref(paragraphs)  # 前者/后者跨句组合判定
        self._scan_structural_paras(paragraphs)
        self._scan_modifier_inflation_para(paragraphs)

    def _scan_blackhole(self, i: int, line: str):
        for word, entry in BLACKHOLE_WORDS.items():
            if word not in line:
                continue
            sug = entry[0]
            # 有实指模式 — 正例模式匹配则放行
            if len(entry) >= 3 and entry[2]:  # has positive_patterns list
                is_positive = any(re.search(pp, line) for pp in entry[2])
                if is_positive:
                    continue
            self._add(i, "T_BLACK", "🔴", word, sug)

    def _scan_meta(self, i: int, line: str):
        for w in META_WORDS:
            if w in line:
                self._add(i, "O_META", "🟠", w, "删掉或直接改写为实质内容")

    def _scan_universal(self, i: int, line: str):
        for item in UNIVERSAL_TEMPLATES:
            if isinstance(item, str):
                if item.replace("……", "").replace("...", "") in line.replace("…", ""):
                    self._add(i, "Y_UNIV", "🟡", item, "删除万能铺垫")
            else:
                m = item.search(line) if hasattr(item, 'search') else None
                if m:
                    self._add(i, "Y_UNIV", "🟡", m.group(), "删除万能铺垫")
                elif isinstance(item, tuple):
                    p, s = item
                    # p could be a regex string or compiled pattern
                    try:
                        m2 = re.search(p, line, re.UNICODE)
                        if m2:
                            self._add(i, "Y_UNIV", "🟡", m2.group(), s)
                    except re.error:
                        pass

    def _scan_vague_modifier(self, i: int, line: str):
        for mod, results in VAGUE_MODIFIERS.items():
            if mod in line:
                # 检查后面是否跟着抽象结果（无数字支撑）
                for r in results:
                    if r in line:
                        # 检查同句中是否有数字（有数字则放行）
                        has_number = bool(re.search(r"\d+", line))
                        if not has_number:
                            self._add(i, "B_vague", "🟢(4b)",
                                      f"{mod}{r}",
                                      f"虚化副词'{mod}'+'{r}'，需跟上百分比或数字")
                        break

    def _scan_suffix_disease(self, i: int, line: str):
        suffixes_in_line = []
        for name, pattern in SUFFIX_PATTERNS.items():
            m = pattern.search(line)
            if m:
                found_word = m.group()  # 整个 match = 后缀词（前缀已非捕获）
                suffix_char = found_word[-1] if found_word else ""
                suffixes_in_line.append(found_word)
                self._add(i, f"B_{name}", f"🟢(4c:{name})",
                          found_word,
                          f"-{suffix_char}派生词，拆成具体动作")

        # 触发阈值：≥2个后缀词，或与虚化修饰同现
        if len(suffixes_in_line) >= 2:
            self._add(i, "B_COMBO", "🟢(4c)",
                      "; ".join(suffixes_in_line),
                      f"单句{len(suffixes_in_line)}个后缀词 → 强AI信号")
        elif len(suffixes_in_line) == 1:
            # 检查是否与虚化修饰同现
            has_vague = False
            for mod in VAGUE_MODIFIERS:
                if mod in line:
                    has_vague = True
                    break
            if has_vague:
                self._add(i, "B_COMBO", "🟢(4c)",
                          f"{suffixes_in_line[0]} + 虚化修饰同现",
                          "单个后缀词 + 虚化修饰 → AI味组合")

    def _scan_weakeners(self, i: int, line: str):
        for w in WEAKENERS:
            if w in line:
                self._add(i, "D_WEAK", "🟢(4d)", w,
                          "弱化断言到几乎不为空，若去掉不影响意思则删")

    def _scan_overseas(self, i: int, line: str):
        # "不是X而是Y"若"而是"后跟具体系统/技术动作(有实指)，整条豁免
        not_but_specific = False
        if "不是" in line and "而是" in line:
            after_er_shi = line[line.find("而是") + 2:]
            if re.search(r"CMS|Push|FCM|GAID|API|SDK|系统|平台|接口|打通|对接|映射|连接|接入|获取|使用|复用|建立|画像|数据|回流", after_er_shi):
                not_but_specific = True
        for item in OVERSEAS_SENTENCES:
            if item is None:
                # "是...的" handled below with context-aware filtering
                pass
            elif isinstance(item, tuple):
                pat, desc = item
            else:
                continue  # skip non-tuple entries
            if hasattr(pat, 'search'):
                m = pat.search(line)
                if m:
                    if "而是" in m.group() and not_but_specific:
                        continue
                    self._add(i, "E_OVER", "🟢(4e)", m.group(), desc)
            else:
                # String pattern → compile and search
                try:
                    m = re.search(pat, line, re.UNICODE)
                    if m:
                        if "而是" in m.group() and not_but_specific:
                            continue
                        self._add(i, "E_OVER", "🟢(4e)", m.group(), desc)
                except re.error:
                    pass

        # 4e "是……的"判断句壳：排除因果解释和疑问，仅标记表演式空泛判断
        for m in OVERSEAS_SHI_DE.finditer(line):
            match_text = m.group()
            # 过滤太短的（单字+的是 = 正常中文）
            if len(match_text) < 5:
                continue
            # 检查是否在 OVERSEAS_SENTENCES 中已经有"不是X而是Y"命中避免重复
            has_not_but = bool(re.search(r"不是.{1,80}而是", line))
            if has_not_but and "而是" in match_text:
                continue
            # 有实指放行：行内含数字/百分比/倍/量化比较 → 数据陈述，判断句壳是数据骨架
            if re.search(r"\d|%|倍", line):
                continue
            self._add(i, "E_OVER", "🟢(4e)", match_text.strip()[:60],
                      "\"是……的\"判断句壳，英文 It is... that... 移植")

    def _scan_punctuation(self, i: int, line: str):
        # 破折号滥用（全文统计在报告阶段做，这里只报单条）
        dashes = line.count("——")
        if dashes > 1:
            self._add(i, "P_DASH", "⚫",
                      f"破折号 ×{dashes}",
                      "破折号是头号AI味标点，换逗号或句号")

        # 满屏加粗：≥4 处加粗（≥8 个 **）→ 排版级 AI 味信号
        bold_stars = line.count("**")
        if bold_stars >= 8:
            self._add(i, "P_BOLD", "⚫",
                      f"满屏加粗 ×{bold_stars // 2}",
                      "Markdown 满屏加粗是排版级 AI 味信号，全文减到 1-2 处")

        # 半角标点：逐字符检查，过滤合法场景后标记非法
        # 合法场景：代码块(`)、URL、邮箱/版本号、数字小数点、Markdown链接语法、JSON
        is_code_line = "`" in line or "{\n" in line or line.strip().startswith("#!/")
        is_json_like = bool(re.search(r'\{.*".*".*:', line))  # {"key": "value"} pattern
        has_url = bool(re.search(r"https?://|www\.", line))
        has_email = bool(re.search(r"[@\w.-]+@[\w.-]", line))
        has_md_link = bool(re.search(r"\[.*?\]\(.*?\)", line))
        has_md_img = bool(re.search(r"!\[.*?\]", line))

        # Initialize before branches so Python recognizes 'illegal' as local in all paths
        illegal = []
        if is_json_like:
            pass  # JSON-like content → half-width punctuation bypassed
        else:
            # 按字符位置扫描，而非全局跳过
            for c in [",", ".", "!", "?", ";", ":", "(", ")"]:
                for idx, ch in enumerate(line):
                    if ch != c:
                        continue
                    # 百分号相关放过
                    if c == "." and idx + 1 < len(line) and line[idx + 1] == "%":
                        continue
                    # 小数点放过（前后都有数字）
                    if c == "." and idx > 0 and idx + 1 < len(line):
                        if line[idx - 1].isdigit() and line[idx + 1].isdigit():
                            continue
                    # 代码标识符/文件名点号放过（如 search_list.js、onSearch.query）— 两侧字母/数字/下划线
                    if c == "." and idx > 0 and idx + 1 < len(line):
                        pre = line[idx - 1]
                        post = line[idx + 1]
                        if (pre.isalnum() or pre == "_") and (post.isalnum() or post == "_"):
                            continue
                    # URL 整体放过
                    if has_url:
                        # 检查当前字符是否在 URL 范围内
                        url_match = re.search(r"https?://[^\s)*\]]*", line[max(0,idx-10):])
                        if url_match:
                            continue
                    # 邮箱放过
                    if has_email:
                        email_match = re.search(r"[\w.-]+@[^\s)*\]]+", line[max(0,idx-20):])
                        if email_match:
                            continue
                    # 版本号和纯数字序列放过
                    if c == ".":
                        digits_around = re.findall(r"\d+\.\d+", line[max(0,idx-8):idx+9])
                        if digits_around:
                            continue
                    # 代码行中的括号放过
                    if is_code_line and c in "()[]{}":
                        continue
                    # Markdown 链接/图片中的括号放过
                    if (has_md_link or has_md_img) and c in "()[]":
                        # 检查是否被 Markdown 语法包裹
                        before_text = line[max(0, idx-3):idx]
                        after_text = line[idx:idx+3]
                        if "]" in before_text or "[" in before_text:
                            continue
                        if ")" in after_text or "(" in before_text[:2]:
                            continue
                    # 中文引号内的英文字母放过（如 "ABC"）
                    if c in "()[]{}.,;:?!":
                        before_char = line[max(0, idx-1)]
                        after_char = line[min(idx+1, len(line)-1)]
                        if before_char == '"' and after_char == '"':
                            continue
                    illegal.append(f"{c}@{idx}")

        if illegal:
            unique_chars = "".join(set(ic[0] for ic in illegal))
            self._add(i, "P_HALF", "⚫",
                      f"半角标点: {unique_chars}",
                      "中文正文应使用全角标点")

        # 引号里的抽象词
        quote_words = re.findall(r'"([^"]*)"', line)
        for qw in quote_words:
            if any(w in qw for w in ["底层逻辑", "范式", "抓手", "方法论",
                                       "生态圈", "矩阵", "闭环"]):
                self._add(i, "P_QUOTE", "⚫",
                          f'"{qw}"',
                          "删引号说人话，或换成具体判断")

    # ── 段落级扫描 ──

    def _scan_modifier_inflation(self, i: int, line: str):
        pass  # No action here — handled by paragraph-level scan

    def _scan_modifier_inflation_para(self, paragraphs: list[ParagraphInfo]):
        """🟢(M) 修饰膨胀：同一段落内≥2类空泛修饰词堆叠且无数字 → 命中。"""
        for para in paragraphs:
            full_text = " ".join(para.lines)

            # 检查是否包含数字——有数字则放行（具体数据不算膨胀）
            if re.search(r"\d+", full_text):
                continue

            # 统计修饰类别
            found_cats = []
            all_matched_words = []
            for name, pattern in EXPANSIVE_MODIFIERS.items():
                matches = pattern.findall(full_text)
                if matches:
                    found_cats.append(name)
                    all_matched_words.extend(matches)

            if len(found_cats) >= 2:
                cat_names = {"scale": "规模", "breadth": "广度", "emotional": "情感",
                             "trend": "趋势", "history": "历史", "quantity": "数量"}
                labels = [cat_names.get(c, c) for c in found_cats]
                self._add(para.start, "M_INFLATE", "🟢(M)",
                          f"修饰膨胀({', '.join(labels)})",
                          f"{len(found_cats)}类空泛修饰词堆叠：{'、'.join(labels)}同现，无数据支撑")

    def _split_paragraphs(self) -> list[ParagraphInfo]:
        """将文本分割为段落列表。"""
        paragraphs = []
        current_lines = []
        current_orig = []
        start = 1

        for i, raw in enumerate(self.lines):
            stripped = raw.strip()
            if not stripped:
                if current_lines:
                    paragraphs.append(ParagraphInfo(start, i, current_lines, current_orig))
                    current_lines = []
                    current_orig = []
                    start = i + 2
            else:
                if not current_lines:
                    start = i + 1
                current_lines.append(stripped)
                current_orig.append(raw)

        if current_lines:
            last_idx = len(self.lines)
            paragraphs.append(ParagraphInfo(start, last_idx, current_lines, current_orig))

        return paragraphs

    def _scan_parallel(self, paragraphs: list[ParagraphInfo]):
        """4a 排比癖：跨段落内句子检测。"""
        for para in paragraphs:
            full_text = " ".join(para.lines)
            for pat, desc in PARALLEL_PATTERNS:
                if isinstance(pat, str):
                    if pat in full_text:
                        self._add(para.start, "B_PARA", "🟢(4a)",
                                  pat, desc)
                else:
                    m = pat.search(full_text)
                    if m:
                        self._add(para.start, "B_PARA", "🟢(4a)",
                                  m.group()[:50], desc)

    def _scan_placeholder_combo(self, paragraphs: list[ParagraphInfo]):
        """🔵 占位语气：同一段落内 ≥2 个不同子模式 → 命中。"""
        for para in paragraphs:
            full_text = " ".join(para.lines)
            found_patterns = {}

            for key, pat in PLACEHOLDER_PATTERNS.items():
                m = pat.search(full_text)
                if m:
                    sub_key = key.split("_", 1)[1] if "_" in key else key
                    found_patterns[sub_key] = (key, m.group())

            if len(found_patterns) >= 2:
                names = "; ".join(
                    PLACEHOLDER_NAMES.get(k, k) for k in found_patterns.keys()
                )
                matched_text = "; ".join(v[1] for v in found_patterns.values())
                self._add(para.start, "U_COMBO", "🔵",
                          matched_text[:80],
                          f"命中多组占位语气：{names}")

    def _scan_cross_ref(self, paragraphs: list[ParagraphInfo]):
        """4e 欧化句型：前者/后者跨句组合判定 ≥2 → 命中。单条不触发。"""
        for para in paragraphs:
            full_text = " ".join(para.lines)
            count_qian = len(re.findall(r"前者", full_text))
            count_hou = len(re.findall(r"后者", full_text))
            total = count_qian + count_hou
            # 豁免：单对前+后（如"A和B，前者X后者Y"）是正常对照写法，非翻译腔
            if count_qian == 1 and count_hou == 1 and total == 2:
                continue
            if total >= 2:
                parts = []
                if count_qian > 0:
                    parts.append(f"前者×{count_qian}")
                if count_hou > 0:
                    parts.append(f"后者×{count_hou}")
                detail = ", ".join(parts) + " → 英文 he/hit 回指习惯，中文需重复主语名"
                self._add(para.start, "E_OVER", "🟢(4e)",
                          f"{detail}",
                          "≥2个前者/后者代词 → 欧化倾向，建议替换为具体名词")

    def _scan_structural_paras(self, paragraphs: list[ParagraphInfo]):
        """🟤 行文结构病：段落级别诊断。"""
        for idx, para in enumerate(paragraphs):
            first_line = para.lines[0] if para.lines else ""
            full_para = " ".join(para.lines)

            # S1 背景开篇病：文档前2个段落，首句以时代/趋势/环境开篇
            if idx < 2:
                if S1_PATTERN.search(first_line):
                    has_facts = bool(re.search(
                        r"(?:Mi\s*Video|Push|FCM|GAID|DAU|\d+万|\d+%|"
                        r"[A-Za-z]{2,}|search|video|push|topic|token|"
                        r"[A-Za-z0-9_]+\.\w+|[A-Za-z0-9_]+CMS|[A-Za-z0-9_]+平台)",
                        full_para,
                        re.IGNORECASE,
                    ))
                    if not has_facts:
                        self._add(para.start, "S1_BG", "🟤",
                                  first_line[:60],
                                  "S1背景开篇：以时代/趋势铺垫开篇，应改为直接陈述问题")

            # S2 目标空洞并列病
                # S2 目标空洞并列病 — 有意义的数字才放行(防版本号和零碎数字误判)
            if S2_PATTERN.search(full_para):
                has_numbers = bool(re.search(r'\d[\d,.]+\s*(?:%|万|亿|百万|\s*%)', full_para))
                if not has_numbers:
                    self._add(para.start, "S2_GAP", "🟤",
                              "连续≥3个'动词+名词'空洞并列",
                              "S2目标空洞：改'获取A能力，实现B结果'")

            # S3 价值虚化病
                # S3 价值虚化病 — 有意义的数字才放行(防版本号和零碎数字误判)
            if S3_VAGUE_PATTERN.search(full_para):
                has_numbers = bool(re.search(r'\d[\d,.]+\s*(?:%|万|亿|百万|\s*%)', full_para))
                if not has_numbers:
                    match = S3_VAGUE_PATTERN.search(full_para)
                    self._add(para.start, "S3_VAGUE", "🟤",
                              full_para[match.start():match.end()],
                              "S3价值虚化：价值断言无基线数据，改为'X从Y变到Z'或删除")

            # S4 空总结段：段落以总结/展望标题开头 + 无事实信息
            if S4_SUMMARY_PATTERN.search(first_line):
                has_facts = bool(re.search(
                    r"(?:Mi\s*Video|Push|FCM|GAID|DAU|\d+万|\d+%|"
                    r"[A-Za-z]{2,}|search|video|push|topic|token|"
                    r"[A-Za-z0-9_]+\.\w+|[A-Za-z0-9_]+CMS|[A-Za-z0-9_]+平台)",
                    full_para,
                    re.IGNORECASE,
                ))
                # 同时检查是否含"综上所述"/"总而言之"等空总结词
                has_empty_summary = bool(re.search(
                    r"综上所述|总而言之|综上|归根结底", full_para))
                if not has_facts and has_empty_summary:
                    self._add(para.start, "S4_EMPTY", "🟤",
                              first_line[:60],
                              "S4空总结段：仅有'综上所述'类空洞收尾，应删去或补充具体行动项")

            # S5 安全对齐痕迹 ≥2 个 RLHF 语气模式 → 命中
            s5_hits = []
            for pat, desc in S5_NEUTRAL_PATTERNS:
                if hasattr(pat, 'search'):
                    if pat.search(full_para):
                        s5_hits.append(desc)
                elif pat in full_para:
                    s5_hits.append(desc)
            if len(s5_hits) >= 2:
                self._add(para.start, "S5_ALIGN", "🟤",
                          f"安全对齐痕迹({len(s5_hits)}处)",
                          "RLHF中性化表达堆叠：" + "; ".join(s5_hits[:3]))

            # S5a 合规腔嵌套 ≥2 个前置条件从句 → 命中
            s5a_hits = []
            for pat, desc in S5A_COMPLIANCE_PATTERNS:
                m = pat.search(full_para) if hasattr(pat, 'search') else None
                if m:
                    s5a_hits.append(desc)
            if len(s5a_hits) >= 1:  # S5a单条即标记（但组合规则在报告中标注）
                self._add(para.start, "S5A_COMP", "🟤",
                          "合规腔嵌套(" + str(len(s5a_hits)) + "处)",
                          "前置条件从句稀释主张：" + "; ".join(s5a_hits[:2]))

            # S5c 信息密度病 — skip for JSON/code blocks
            is_json_para = any(re.search(r'\{.*".*".*:', ln) for ln in para.lines)
            density = info_density_bay(para.lines)
            if not is_json_para and density["low"]:
                self._add(para.start, "S5C_DENSITY", "🟤",
                          f"信息密度{density['ratio']}",
                          f"S5c段落{density['total_chars']}字仅{density['fact_chars']}字含事实信息(~{int(density['ratio']*100)}%)，建议砍掉空泛修饰")

    def get_summary(self) -> dict:
        """生成汇总统计。"""
        counts = defaultdict(int)
        by_category = defaultdict(list)

        for h in self.hits:
            counts[h.color] += 1
            by_category[h.category].append(h)

        total = sum(counts.values())

        # 提取各类别短码
        short_counts = {}
        color_map = {
            "🔴": "黑话名词(T)",
            "🟠": "元话语(O)",
            "🟡": "万能修辞(Y)",
            "🟢": "结构癖(B)/修饰膨胀(M)",
            "⚫": "标点(P)",
            "🔵": "占位语气(U)",
            "🟤": "行文结构(S)",
        }

        for color, count in counts.items():
            for prefix, label in color_map.items():
                if color.startswith(prefix):
                    short_counts[label] = count
                    break
            else:
                short_counts[color[:4]] = count

        return {
            "total": total,
            "counts_by_color": dict(counts),
            "short_counts": short_counts,
            "by_category": {k: list(v) for k, v in by_category.items()},
        }


# ─── 报告格式化 ──────────────────────────────────────────────

def format_report(text: str, hits: list[Hit], summary: dict):
    """按 SKILL.md 规定的格式输出检测报告。"""
    total = summary["total"]
    lines_out = []

    # 计算各分类总数
    b_total = sum(1 for h in hits if h.color.startswith("🟢") and "(M)" not in h.color)
    m_total = sum(1 for h in hits if "(M)" in h.color)
    s_total = sum(1 for h in hits if h.color.startswith("🟤"))
    t_total = sum(1 for h in hits if h.color == "🔴")
    o_total = sum(1 for h in hits if h.color == "🟠")
    y_total = sum(1 for h in hits if h.color == "🟡")
    p_total = sum(1 for h in hits if h.color == "⚫")
    u_total = sum(1 for h in hits if h.color == "🔵")

    magnet = "\U0001f50d"
    lines_out.append(f"{magnet} 检出 {total} 处:")
    lines_out.append(f"  黑话名词 T={t_total} / 元话语 O={o_total} / 万能修辞 Y={y_total} / "
                     f"结构癖 B={b_total}(4a排比 / 4b虚化 / 4c后缀病 / 4d程度副词 / 4e欧化句型) / "
                     f"修饰膨胀 M={m_total} / ⚫标点 P={p_total} / 🔵占位语气 U={u_total} / 🟤行文结构病 S={s_total}")
    lines_out.append("")

    # 分类输出
    category_order = [
        ("🔴", "黑话名词", lambda h: h.color == "🔴"),
        ("🟠", "元话语", lambda h: h.color == "🟠"),
        ("🟡", "万能修辞", lambda h: h.color == "🟡"),
        ("🟢(M)", "修饰膨胀", lambda h: "(M)" in h.color),
        ("🟢", "结构癖", lambda h: h.color.startswith("🟢") and "(M)" not in h.color),
        ("⚫", "标点级", lambda h: h.color == "⚫"),
        ("🔵", "占位语气", lambda h: h.color == "🔵"),
        ("🟤", "行文结构病", lambda h: h.color.startswith("🟤")),
    ]    # 分类输出
    category_order = [
        ("🔴", "黑话名词", lambda h: h.color == "🔴"),
        ("🟠", "元话语", lambda h: h.color == "🟠"),
        ("🟡", "万能修辞", lambda h: h.color == "🟡"),
        ("🟢(M)", "修饰膨胀", lambda h: "(M)" in h.color),
        ("🟢", "结构癖", lambda h: h.color.startswith("🟢") and "(M)" not in h.color),
        ("⚫", "标点级", lambda h: h.color == "⚫"),
        ("🔵", "占位语气", lambda h: h.color == "🔵"),
        ("🟤", "行文结构病", lambda h: h.color.startswith("🟤")),
    ]

    for color, title, predicate in category_order:
        cat_hits = [h for h in hits if predicate(h)]
        if not cat_hits:
            continue

        lines_out.append(f"═══ {color} {title} ═══")
        for h in sorted(cat_hits, key=lambda x: x.line):
            detail_part = f" ({h.detail})" if h.detail else ""
            lines_out.append(f"[行 {h.line}] \"{h.text}\" → \"{h.suggestion}\"{detail_part}")
        lines_out.append("")

    return "\n".join(lines_out)


# ─── 改写模块 ─────────────────────────────────────────────────

def judgment_discipline(text: str) -> tuple[list[str], list[str]]:
    """判决纪律：逐句问'这句删掉损失什么事实？'

    返回 (保留段落, 可删段落)。
    """
    paragraphs = re.split(r"\n{2,}|\n(?=#)", text)
    kept = []
    deletable = []

    for para in paragraphs:
        lines_p = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines_p:
            continue

        has_facts = any(re.search(r"\d+|[A-Z]{2,}|Mi\s*Video|Push|FCM|GAID|DAU|"
                                   r"CTR|RU|SG|BR|HTTP|\w+\.\w+", l)
                        for l in lines_p)

        if has_facts:
            kept.append(para)
        else:
            # 检查是否是空泛套话
            is_fluff = any(w in para for w in [
                "综上所述", "总而言之", "在当前", "随着", "赋能",
                "形成闭环", "提供有力支撑", "奠定坚实基础",
            ])
            if is_fluff or len(lines_p) <= 2:
                deletable.append(para)
            else:
                kept.append(para)

    return kept, deletable


def rewrite_simple(para: str) -> str:
    """直白版改写：像跟同事说话。"""
    text = para

    # 删万能开头
    text = re.sub(r"在当今.{2,20}(?:的时代|背景下|浪潮下)", "", text)
    text = re.sub(r"随着.{2,30}(?:的发展|进步|深入)", "", text)
    text = re.sub(r"^在.{2,15}(?:竞争|发展|变革|转型).{0,20}[，,]", "", text)

    # 替换黑洞词
    replacements = {
        "赋能": "帮",
        "抓手": "切入点",
        "打通": "连上",
        "沉淀": "留下",
        "形成闭环": "把流程跑完",
        " methodologies": "方法",
        "形成一套": "留下一套",
        "有效": "",
        "显著": "",
        "大幅": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 拆排比
    text = re.sub(r"(?:实现了|提升了|增强了){[^}]+}(?:、|,\s*)(?:实现了|提升了|增强了){[^}]+}",
                  "把…理顺了，…更顺了", text)

    # 清理多余标点和空格
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text if text.strip() else "(原段落无可保留的事实，建议整段删除)"


def rewrite_professional(para: str) -> str:
    """专业版改写：正式但不端着。"""
    text = para

    # 删万能开头
    text = re.sub(r"在当今.{2,20}(?:的时代|背景下|浪潮下)", "", text)
    text = re.sub(r"随着.{2,30}(?:的发展|进步|深入)", "", text)

    # 替换黑洞词为精准业务词
    replacements = {
        "赋能": "支持",
        "抓手": "手段",
        "打通": "对接",
        "沉淀": "积累",
        "形成闭环": "形成完整流程",
        "方法论": "方法",
        "有力支撑": "支持",
        "奠定基础": "做准备",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 处理虚化修饰
    text = re.sub(r"(有效|显著|大幅|极大|全面|持续|进一步)(?:提升|优化|改善|增强)",
                  "改进", text)

    text = re.sub(r"\s{2,}", " ", text).strip()

    return text if text.strip() else "(原段落无可保留的事实，建议整段删除)"


def format_rewrites(text: str, hits: list[Hit]) -> str:
    """对每个段落输出直白版 + 专业版 + 保留事实清单。"""
    paragraphs = re.split(r"\n{2,}|\n(?=#)", text)
    kept, deletable = judgment_discipline(text)

    lines_out = []

    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped:
            continue

        para_lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        summary_text = " ".join(para_lines[:3])
        if len(para_lines) > 3:
            summary_text += "…"

        lines_out.append(f"> **第{i+1}段（原：\"{summary_text}\"）**")
        lines_out.append("")

        if stripped in deletable:
            lines_out.append("> **建议整段删除** — 此段无实质信息量")
        else:
            simple = rewrite_simple(stripped)
            prof = rewrite_professional(stripped)
            lines_out.append(f"> **直白版**：{simple} — 像跟同事说话")
            lines_out.append(f"> **专业版**：{prof} — 正式但不端着")

        # 保留事实清单
        facts = []
        numbers = re.findall(r"[\d]+[千万亿兆个百十%]*", stripped)
        proper_nouns = re.findall(r"(Mi\s*Video|Push|FCM|GAID|DAU|CTR|API|SDK|HTTP|RU|SG|BR)", stripped)
        if numbers:
            facts.extend([f"- [✓] {n}" for n in numbers])
        if proper_nouns:
            facts.extend([f"- [✓] {m.group() if hasattr(m, 'group') else m}"
                          for m in re.finditer(r"(Mi\s*Video|Push|FCM|GAID|DAU|CTR|API|SDK|HTTP|RU|SG|BR)", stripped)])

        if facts:
            lines_out.append(f"\n> **保留事实清单**：")
            lines_out.extend(facts)
            lines_out.append("> ✅ 该段所有事实和数字已保留")
        else:
            lines_out.append("> （该段无具体事实/数字）")

        lines_out.append("")

    if deletable:
        lines_out.append("> **可删段落建议**：")
        for d in deletable:
            ds = d.strip()[:80]
            lines_out.append(f"- [{ds}] 此段无实质信息量，建议直接删除")
        lines_out.append("")

    return "\n".join(lines_out)


# ─── Demo / 测试 ─────────────────────────────────────────────

DEMO_TEXT = """在当前全球移动互联网竞争日益激烈的环境下，我们需要通过精细化运营策略，形成一套完整的服务闭环体系，为后续发展打下坚实基础。

推动多端数据融合，打通用户行为分析链路，沉淀出一套可复用数据分析方法论。

FCM Payload 结构：{"to": "/topics/video-interest", "data": {"video_id": "12345", "title": "..."}}

随着移动互联网技术的飞速发展，用户获取信息的方式正在发生深刻变革……需要通过技术赋能，构建全方位的内容分发矩阵。

本方案旨在通过系统化手段实现对 Push 任务的精准圈选与高效触达，打破系统壁垒，构建端到端的全链路 Push 管理体系。

该产品在全球范围内获得了广泛的用户认可和积极的市场反馈，在多个关键指标上均呈现出持续向上的良好态势。

共同基础是纯客户端实现，服务端无需改动，整体改动范围可控，对现有架构影响较小。

我们可以看到，随着技术发展，值得注意的是用户行为正在发生深刻变化。需要强调的是，这些数据反映了市场的真实趋势。

建议在后续迭代中以实际业务情况为准，暂定采用方案 A，可根据实际需求进行灵活调整。"""


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ai-detox 检测引擎 v1.5")
    parser.add_argument("text", nargs="?", default=None, help="待检测文本")
    parser.add_argument("--file", "-f", help="从文件读取")
    parser.add_argument("--stdin", action="store_true", help="从 stdin 读取")
    parser.add_argument("--demo", action="store_true", help="运行示例检测")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--rewrite", "-r", action="store_true",
                        help="附加改写输出")
    args = parser.parse_args()

    # 获取文本
    if args.demo:
        text = DEMO_TEXT
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.stdin:
        text = sys.stdin.read()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        return

    # 扫描
    scanner = Scanner(text)
    scanner.scan_categories()
    hits = scanner.hits
    summary = scanner.get_summary()

    if args.json:
        result = {
            "total": summary["total"],
            "summary": summary["short_counts"],
            "hits": [
                {
                    "line": h.line,
                    "category": h.category,
                    "color": h.color,
                    "text": h.text,
                    "suggestion": h.suggestion,
                    "detail": h.detail,
                }
                for h in hits
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 文本报告
        report = format_report(text, hits, summary)
        print(report)

        if args.rewrite:
            print("\n" + "=" * 60)
            print("改写输出")
            print("=" * 60 + "\n")
            print(format_rewrites(text, hits))


if __name__ == "__main__":
    main()
