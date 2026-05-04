"""构建 CCEMA 项目的中文研究汇报幻灯片(简体中文版本)。

每张幻灯片由一张 matplotlib 图(16:9, 13.33 x 7.5 英寸)渲染,然后编译为:
  - PDF (matplotlib.backends.backend_pdf.PdfPages)
  - PPTX (python-pptx,每张幻灯片以满版图片插入)

用法:
    python scripts/build_report_ppt_zh.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

from pptx import Presentation
from pptx.util import Inches

# ---------------------------------------------------------------------------
# CJK font configuration (macOS)
# ---------------------------------------------------------------------------
# Note: on macOS the system PingFang.ttc collection registers `PingFang HK`
# (Hong Kong / Traditional variant) with matplotlib's font_manager, but the
# HK subfont LACKS several simplified-only glyphs (e.g. 稳, 杂). To keep the
# user-requested PingFang HK as the primary face while still rendering all
# simplified characters, we extract the PingFang SC subfont (index 3 of the
# collection) once into a cache directory and register it explicitly. The
# fallback chain then becomes: PingFang HK -> PingFang SC -> Songti SC ->
# Heiti TC -> Hiragino Sans GB -> DejaVu Sans.
_PINGFANG_TTC = "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc"
_FONT_CACHE = Path.home() / ".cache" / "ccema_zh_fonts"
_FONT_CACHE.mkdir(parents=True, exist_ok=True)
_PINGFANG_SC = _FONT_CACHE / "PingFangSC-Regular.ttf"

if not _PINGFANG_SC.exists() and Path(_PINGFANG_TTC).exists():
    try:
        from fontTools.ttLib import TTCollection
        _ttc = TTCollection(_PINGFANG_TTC)
        # Index 3 in the collection is the PingFang SC Regular face.
        _ttc.fonts[3].save(str(_PINGFANG_SC))
    except Exception as _exc:  # noqa: BLE001
        print(f"[font_manager] could not extract PingFang SC: {_exc}")

if _PINGFANG_SC.exists():
    try:
        font_manager.fontManager.addfont(str(_PINGFANG_SC))
    except Exception as _exc:  # noqa: BLE001
        print(f"[font_manager] could not register PingFang SC: {_exc}")

_CJK_CANDIDATES = ["PingFang HK", "PingFang SC", "Songti SC", "Heiti TC", "Hiragino Sans GB", "STHeiti"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_chosen = next((f for f in _CJK_CANDIDATES if f in _available), None)
if _chosen is None:
    _chosen = "DejaVu Sans"
    print(f"[font_manager] WARNING: no CJK font found among {_CJK_CANDIDATES}; falling back to DejaVu Sans")
else:
    print(f"[font_manager] using {_chosen}")

# Build font family list with the chosen CJK font first, then all available
# CJK fallbacks (so missing glyphs in PingFang HK fall through to PingFang SC).
_font_family = [_chosen] + [f for f in _CJK_CANDIDATES if f != _chosen and f in _available] + ["DejaVu Sans"]
plt.rcParams["font.family"] = _font_family
plt.rcParams["axes.unicode_minus"] = False
# Ensure mathtext keeps standard rendering (math stays English/LaTeX)
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "outputs" / "ppt" / "figures"
OUT_DIR = REPO_ROOT / "docs" / "presentations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "ccema_report_v1_zh.pdf"
PPTX_PATH = OUT_DIR / "ccema_report_v1_zh.pptx"

# ---------------------------------------------------------------------------
# Page layout constants
# ---------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 13.33, 7.5  # inches (16:9)
TITLE_Y = 0.93
SUBTITLE_Y = 0.86

C_BG = "#ffffff"
C_TITLE = "#0b3d91"
C_ACCENT = "#c0392b"
C_SUB = "#2c3e50"
C_BODY = "#1a1a1a"
C_GREY = "#7f8c8d"
C_LIGHT = "#ecf0f1"

PLACEHOLDERS_USED: list[str] = []

TOTAL_SLIDES = 32


# ---------------------------------------------------------------------------
# Slide-frame helpers
# ---------------------------------------------------------------------------
def new_slide(title: str | None = None, subtitle: str | None = None):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Top accent bar
    ax.add_patch(Rectangle((0, 0.985), 1, 0.015, color=C_TITLE, transform=ax.transAxes))
    # Bottom accent bar
    ax.add_patch(Rectangle((0, 0), 1, 0.015, color=C_TITLE, transform=ax.transAxes))

    if title is not None:
        ax.text(0.04, TITLE_Y, title, fontsize=24, color=C_TITLE,
                fontweight="bold", va="center", ha="left")
    if subtitle is not None:
        ax.text(0.04, SUBTITLE_Y, subtitle, fontsize=13, color=C_SUB,
                style="italic", va="center", ha="left")
    return fig, ax


def footer(ax, slide_num: int, total: int = TOTAL_SLIDES):
    ax.text(0.04, 0.025, "CCEMA 研究汇报 · v1 · 2026-05-04",
            fontsize=8, color=C_GREY, va="center", ha="left")
    ax.text(0.96, 0.025, f"第 {slide_num} / {total} 页",
            fontsize=8, color=C_GREY, va="center", ha="right")


def bullet_list(ax, items, x=0.06, y0=0.78, dy=0.06, fontsize=14,
                color=C_BODY, bullet="·"):
    y = y0
    for it in items:
        ax.text(x, y, f"{bullet}  {it}", fontsize=fontsize, color=color,
                va="top", ha="left", wrap=True)
        y -= dy


def math_lines(ax, lines, x=0.08, y0=0.72, dy=0.075, fontsize=15,
               color=C_BODY):
    y = y0
    for ln in lines:
        ax.text(x, y, ln, fontsize=fontsize, color=color,
                va="top", ha="left")
        y -= dy


def embed_figure(ax, fig_name: str, bbox=(0.30, 0.10, 0.66, 0.66)):
    """从 outputs/ppt/figures 嵌入 PNG。若缺失则渲染占位框。

    bbox = (x0, y0, width, height),取幻灯片归一化坐标。
    """
    png_path = FIG_DIR / fig_name
    x0, y0, w, h = bbox
    if png_path.exists() and png_path.stat().st_size > 0:
        try:
            img = mpimg.imread(str(png_path))
            inset = ax.figure.add_axes([x0, y0, w, h])
            inset.imshow(img)
            inset.set_axis_off()
            return True
        except Exception as exc:  # noqa: BLE001
            PLACEHOLDERS_USED.append(f"{fig_name} (read error: {exc})")
    else:
        PLACEHOLDERS_USED.append(fig_name)

    # Placeholder
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=C_LIGHT,
                           edgecolor=C_GREY, linewidth=1, linestyle="--"))
    ax.text(x0 + w / 2, y0 + h / 2,
            f"[图片占位: {fig_name}]",
            fontsize=14, color=C_GREY, ha="center", va="center", style="italic")
    return False


def section_divider(title, subtitle=""):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_TITLE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.5, 0.55, title, fontsize=44, color="white",
            fontweight="bold", ha="center", va="center")
    if subtitle:
        ax.text(0.5, 0.42, subtitle, fontsize=18, color="#bdd7ff",
                ha="center", va="center", style="italic")
    return fig, ax


# ---------------------------------------------------------------------------
# Individual slides
# ---------------------------------------------------------------------------
def slide_01_title():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Decorative left bar
    ax.add_patch(Rectangle((0, 0), 0.06, 1, color=C_TITLE))
    ax.add_patch(Rectangle((0.06, 0), 0.005, 1, color=C_ACCENT))

    ax.text(0.10, 0.74,
            "面向医疗 AI Agent 的",
            fontsize=34, color=C_TITLE, fontweight="bold", va="center")
    ax.text(0.10, 0.66,
            "反事实对话模拟 (CCEMA)",
            fontsize=34, color=C_TITLE, fontweight="bold", va="center")

    ax.text(0.10, 0.55,
            "面向临床对话 Agent 的离策略评估框架",
            fontsize=18, color=C_SUB, style="italic", va="center")

    ax.text(0.10, 0.42,
            "现代 OPE  +  多智能体辩论  +  因果对比嵌入  +  MSM 敏感度分析",
            fontsize=14, color=C_ACCENT, va="center")

    ax.text(0.10, 0.30, "郝林浩 (Linhao Hao)", fontsize=16, color=C_BODY, va="center")
    ax.text(0.10, 0.25, "lhhao0430@gmail.com", fontsize=12, color=C_GREY, va="center")
    ax.text(0.10, 0.18, "日期:2026-05-04", fontsize=12, color=C_GREY, va="center")
    ax.text(0.10, 0.13, "目标会议/期刊:NeurIPS 2026 / JAMIA / npj Digital Medicine",
            fontsize=12, color=C_GREY, va="center")
    return fig


def slide_02_eval_gap():
    fig, ax = new_slide(
        "医疗 AI 评估的现实缺口",
        "为什么现行评估流程无法回答“能否上线部署”这一问题",
    )
    bullet_list(
        ax,
        [
            "能力 ≠ 部署质量。MedQA / MMLU-Med 准确率已饱和到 90% 以上,但完全无法说明\n   一个 Agent 在真实多轮临床对话中的表现。",
            "前瞻性 RCT 太慢。一次为期 6-12 个月的“静默影子”试验花费超过 100 万美元,\n   且每个站点都需要走 IRB 流程;而 ML 团队每 8-12 周就发布一版新 Agent。",
            "Agent 大约每 3 个月迭代一次。当 RCT 报告出炉时,生产环境模型已经领先 2-3 个版本,\n   结论一发表即过时。",
            "我们需要的是:仅依赖日志的、回顾性的、反事实估计器,带可校准的置信区间,\n   且能在每次发布时重跑而无需新增患者接触。",
        ],
        y0=0.74, dy=0.13, fontsize=13,
    )
    embed_figure(ax, "fig01_problem_overview.png", bbox=(0.05, 0.06, 0.90, 0.18))
    footer(ax, 2)
    return fig


def slide_03_ope_tte_gap():
    fig, ax = new_slide(
        "OPE 与 TTE 之间的鸿沟",
        "两套成熟方法论,但都不适用于以自然语言为动作的医疗 Agent",
    )
    ax.text(0.06, 0.78, "离策略评估 (OPE):", fontsize=15,
            color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "在表格化 / 离散动作场景(广告、推荐系统)中已经成熟。",
            "把动作视为离散类别 id;假设倾向得分已知,或可在小动作空间上估计。",
            "对自由文本动作直接失效:一份临床记录的 |A| ~ 10^1000,倾向得分几乎处处为零,\n   IPS 方差爆炸。",
        ],
        y0=0.74, dy=0.055, fontsize=12,
    )
    ax.text(0.06, 0.45, "目标试验仿真 (TTE):", fontsize=15,
            color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "EHR 数据上因果推断的金标准——在观测数据上仿真一项假想 RCT。",
            "为二值 / 离散治疗设计(用药 vs. 不用药);无法把“重写鉴别诊断段落”\n   编码为一个治疗组。",
            "缺乏可靠机制将由临床医生策略 π_b 推广到任意 LLM 策略 π_e。",
        ],
        y0=0.41, dy=0.055, fontsize=12,
    )
    ax.text(0.06, 0.12,
            "CCEMA 弥合二者之间的鸿沟:TTE 假设 + OPE 估计器 +\n"
            "一个让自由文本动作可处理的因果对比嵌入。",
            fontsize=13, color=C_ACCENT, fontweight="bold")
    footer(ax, 3)
    return fig


def slide_04_setup():
    fig, ax = new_slide(
        "问题设定与符号约定",
        "日志数据、行为策略、评估策略",
    )
    math_lines(
        ax,
        [
            r"日志数据集:  $\mathcal{D} = \{(x_i, a_i^{clin}, y_i)\}_{i=1}^n$",
            r"     $x_i \in \mathcal{X}$    患者上下文(人口学、病史、对话已发生部分)",
            r"     $a_i^{clin} \in \mathcal{A}$    医生的自由文本动作(病历、诊疗计划、回复)",
            r"     $y_i \in [0,1]$    实际产生的结局(医学评审员的过程奖励分)",
            "",
            r"行为策略:    $\pi_b(a \mid x) \;=\;$ 未知的临床医生在 $\mathcal{A}$ 上的分布",
            r"评估策略:  $\pi_e(a \mid x) \;=\;$ 候选 LLM Agent(由我方控制)",
            "",
            r"目标估计量:    $V(\pi_e) \;=\; \mathbb{E}_{x \sim p(x)} [\, \mathbb{E}_{a \sim \pi_e(\cdot\mid x)}[\, Y(x,a) \,] \,]$",
            "",
            r"“评测台”要回答的问题:仅由 $\mathcal{D}$ 估计 $V(\pi_e)$,并给出可校准的 CI。",
        ],
        x=0.07, y0=0.78, dy=0.065, fontsize=14,
    )
    footer(ax, 4)
    return fig


def slide_05_estimand():
    fig, ax = new_slide(
        "估计目标:V(π_e) 与重要性采样恒等式",
        "完整的 IS 推导,逐行展开",
    )
    math_lines(
        ax,
        [
            r"第 1 步——定义:",
            r"     $V(\pi_e) \;=\; \int p(x) \int \pi_e(a\mid x)\, \mathbb{E}[Y\mid x,a]\, da\, dx$",
            "",
            r"第 2 步——同时乘除以行为分布密度(正性条件,见下页):",
            r"     $V(\pi_e) \;=\; \int p(x) \int \pi_b(a\mid x)\, \frac{\pi_e(a\mid x)}{\pi_b(a\mid x)}\, \mathbb{E}[Y\mid x,a]\, da\, dx$",
            "",
            r"第 3 步——识别为日志分布 $p(x)\pi_b(a\mid x)$ 下的期望:",
            r"     $V(\pi_e) \;=\; \mathbb{E}_{(x,a) \sim \pi_b}\!\left[\, \frac{\pi_e(a\mid x)}{\pi_b(a\mid x)}\, Y \,\right]$",
            "",
            r"第 4 步——代入(IPS)估计:",
            r"     $\hat V_{IPS} \;=\; \frac{1}{n} \sum_{i=1}^n \frac{\pi_e(a_i\mid x_i)}{\pi_b(a_i\mid x_i)}\, y_i$",
            "",
            r"在正性 + 一致性下无偏;方差按 $\sup_{x,a} \pi_e/\pi_b$ 增长。",
        ],
        x=0.07, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 5)
    return fig


def slide_06_assumptions():
    fig, ax = new_slide(
        "五项识别假设",
        "标准的因果推断公理,改写到自由文本医疗动作的语境下",
    )
    items = [
        ("一致性 (SUTVA-1)",
         r"$Y_i = Y_i(a_i^{clin})$——观测到的结局等于在所采取动作下的潜在结局。"),
        ("正性 / 共同支撑",
         r"$\pi_b(a\mid x) > 0$ 当且仅当 $\pi_e(a\mid x) > 0$——医生在合理范围内可以写出 Agent 可能写出的任何文本。"),
        ("条件可交换性(无未观测混杂)",
         r"$Y(a) \,\perp\!\!\!\perp\, A \mid X$——在给定 $X=x$ 的层内,动作分配相当于随机化。"),
        ("无预期效应",
         r"结局不依赖于本轮之后才采取的动作(排除回顾性日志中的“前瞻偏倚”)。"),
        ("无干扰 (SUTVA-2)",
         r"在给定上下文条件下,$Y_i$ 不依赖于 $a_j$ ($j \ne i$)——病例之间相互独立。"),
    ]
    y = 0.78
    for name, desc in items:
        ax.text(0.06, y, name, fontsize=13, color=C_TITLE, fontweight="bold")
        ax.text(0.06, y - 0.04, desc, fontsize=11.5, color=C_BODY)
        y -= 0.12
    ax.text(0.06, 0.10,
            "U3(因果对比嵌入)通过把自由文本动作压缩到低维、与混杂因子正交的嵌入空间,\n"
            "使得正性与可交换性变得可辩护。",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 6)
    return fig


def slide_07_upgrades_table():
    fig, ax = new_slide(
        "四项升级总览",
        "U1-U4:替换了什么、来源、以及我们的贡献",
    )
    rows = [
        ("U1", "现代 OPE 估计器",
         "朴素 IPS / DM",
         "DML (Chernozhukov 2018)、TMLE (van der Laan 2006)、Conformal CI (Taufiq 2022)",
         "首次落地到临床 NLP;完整的 DR 交叉拟合 + 目标更新实现"),
        ("U2", "多智能体辩论裁判",
         "单一 LLM 作裁判",
         "Khan 2024、Chan 2024(多智能体辩论)",
         "三方:辩护-起诉-裁判,跨厂商多样化 + Beta 后验"),
        ("U3", "因果对比嵌入",
         "现成的 SBERT/MedCPT 编码动作",
         "InfoNCE (Oord 2018)、HSIC (Gretton 2005)、MIPS (Saito 2022)",
         "核心贡献;真实 ACI-Bench 上 AUC=1.0 → 0.55,使 OPE 终于可识别"),
        ("U4", "PSE 分解 + MSM",
         "ΔV 的单点估计",
         "Pearl (2001)、Yadlowsky (2018) MSM",
         "Logistic-MSM 的脆弱度 Γ* 闭式解;四轴路径特异分解"),
    ]
    # Header
    headers = ["", "替换 / 新增内容", "替换对象", "理论来源", "我方贡献"]
    col_x = [0.04, 0.12, 0.30, 0.46, 0.69]
    col_w = [0.05, 0.18, 0.16, 0.23, 0.27]
    y_head = 0.79
    ax.add_patch(Rectangle((0.03, y_head - 0.05), 0.94, 0.05,
                           facecolor=C_TITLE, edgecolor="none"))
    for h, x in zip(headers, col_x):
        ax.text(x, y_head - 0.025, h, fontsize=11, color="white",
                fontweight="bold", va="center")

    y = y_head - 0.10
    for i, row in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.03, y - 0.085), 0.94, 0.10,
                                   facecolor="#f4f6f9", edgecolor="none"))
        ax.text(col_x[0], y, row[0], fontsize=14, color=C_ACCENT,
                fontweight="bold", va="top")
        for j in range(1, 5):
            txt = row[j]
            ax.text(col_x[j], y, txt, fontsize=10, color=C_BODY,
                    va="top", wrap=True)
        y -= 0.13

    ax.text(0.06, 0.07,
            "整体效果:在仅有日志数据的前提下,完成识别 + 估计 + 不确定性 + 敏感度。",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 7)
    return fig


def slide_08_dm_ips_dr():
    fig, ax = new_slide(
        "U1——DM、IPS、DR 回顾",
        "三类经典 OPE 估计器并列对比",
    )
    math_lines(
        ax,
        [
            r"直接法 (DM):  拟合结局模型 $\hat f(x,a) \approx \mathbb{E}[Y\mid x,a]$",
            r"     $\hat V_{DM} \;=\; \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{a\sim\pi_e(\cdot\mid x_i)}\,\hat f(x_i, a)$",
            r"     · 方差小;若 $\hat f$ 设定不准则偏倚大",
            "",
            r"逆倾向加权 (IPS):",
            r"     $\hat V_{IPS} \;=\; \frac{1}{n}\sum_i w_i\, y_i, \quad w_i = \pi_e(a_i\mid x_i)/\pi_b(a_i\mid x_i)$",
            r"     · 无偏;但当 $w_i$ 重尾时方差极大",
            "",
            r"双重稳健 (DR, Robins 1994; Dudik 2011):",
            r"     $\hat V_{DR} \;=\; \frac{1}{n}\sum_i [\, \mathbb{E}_{a\sim\pi_e}\hat f(x_i,a) \;+\; w_i( y_i - \hat f(x_i, a_i)) \,]$",
            r"     · 只要 $\hat f$ 与 $\hat \pi_b$ 之一正确即一致(双重鲁棒性)",
        ],
        x=0.06, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 8)
    return fig


def slide_09_dr_overfitting_bias():
    fig, ax = new_slide(
        "U1——为什么朴素 DR 会失效:一阶过拟合偏倚",
        "在同一份数据上估计干扰参数与价值,误差是 O(1) 而非 O(1/√n)",
    )
    math_lines(
        ax,
        [
            r"分解一个朴素 plug-in DR 估计器的误差:",
            r"     $\hat V_{DR} - V \;=\; ( \mathbb{E}_n - \mathbb{E} )\,\psi_0 \;+\; ( \mathbb{E}_n - \mathbb{E} )(\hat\psi - \psi_0) \;+\; \mathrm{bias}(\hat\psi)$",
            r"                    $\;\;\;$ [随机项 $O(n^{-1/2})$]  $\;\;\;$ [漂移项]  $\;\;\;$ [干扰偏倚]",
            "",
            r"漂移项可以归约为干扰项误差的乘积:",
            r"     $|\, \mathrm{drift}\, | \;\leq\; || \hat f - f_0 ||_{L_2} \,\cdot\, || \hat w - w_0 ||_{L_2}$",
            "",
            r"若不进行样本切分,$\hat f$ 会记忆 $\mathcal{D}$——经验过程项",
            r"$(\mathbb{E}_n - \mathbb{E})\,(\hat f - f_0)$ 不会以参数速率消失。",
            "",
            r"后果:偏倚是 $O(1)$ 量级,渐近 CI 覆盖率失真,点估计有偏。",
            "",
            r"补救:把每个干扰参数所拟合的数据留出(交叉拟合,见下页)。",
        ],
        x=0.06, y0=0.78, dy=0.06, fontsize=13,
    )
    footer(ax, 9)
    return fig


def slide_10_dml():
    fig, ax = new_slide(
        "U1——DML / 交叉拟合 (Chernozhukov 等,2018)",
        "K 折样本切分恢复参数收敛速率",
    )
    math_lines(
        ax,
        [
            r"流程 (K = 5):",
            r"     1. 把 $\mathcal{D}$ 划分为 $K$ 个互斥折 $I_1, \dots, I_K$。",
            r"     2. 对 $k = 1\dots K$:在 $\mathcal{D} \setminus I_k$ 上拟合 $\hat f^{(-k)}, \hat w^{(-k)}$。",
            r"     3. 用留出的干扰参数为折 $I_k$ 打分。",
            r"     4. 求平均:   $\hat V_{DML} \;=\; \frac{1}{n}\sum_{k=1}^K \sum_{i\in I_k}\, \psi\!(z_i;\, \hat f^{(-k)}, \hat w^{(-k)}).$",
            "",
            r"渐近线性证明梗概。在折 $I_k$ 上,$\hat f^{(-k)}$ 与 $\{z_i\}_{i\in I_k}$ 独立,",
            r"由条件 Lindeberg CLT 可得 $(\mathbb{E}_n - \mathbb{E})\,(\hat\psi^{(-k)} - \psi_0) = O_p(n^{-1/2})$。",
            r"再结合乘积速率偏倚 $\|\hat f - f_0\|\cdot \|\hat w - w_0\|$,只要两个干扰参数",
            r"以 $n^{-1/4}$ 收敛,我们就有",
            "",
            r"     $\sqrt{n}\,(\hat V_{DML} - V) \;\rightarrow_d\; \mathcal{N}(0,\, \sigma^2_{\psi})$,",
            "",
            r"其中 $\sigma^2_\psi$ 是半参有效方差 (Newey 1994)。",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 10)
    return fig


def slide_11_tmle():
    fig, ax = new_slide(
        "U1——TMLE (van der Laan & Rubin, 2006)",
        "目标更新达到 Cramer-Rao 半参效率界",
    )
    math_lines(
        ax,
        [
            r"初始结局估计 $\hat f(x,a) \in (0,1)$。定义“聪明协变量”",
            r"     $H(x,a) \;=\; \pi_e(a\mid x)\,/\,\pi_b(a\mid x)$.",
            "",
            r"目标更新(单步 logistic 扰动):",
            r"     $f^{*}(x,a) \;=\; \mathrm{sigmoid}\!(\,\mathrm{logit}\,\hat f(x,a) + \varepsilon \cdot H(x,a)\,)$",
            "",
            r"$\varepsilon$ 通过对 $\{(y_i, H_i)\}$ 的极大似然估计:",
            r"     $\hat\varepsilon \;=\; \arg\max_\varepsilon \sum_i [\, y_i\log f^{*}_\varepsilon + (1-y_i)\log(1-f^{*}_\varepsilon)\,]$",
            "",
            r"由构造,得分方程恒成立:",
            r"     $\frac{1}{n}\sum_i H(x_i,a_i)\,(y_i - f^{*}(x_i,a_i)) \;=\; 0.$",
            "",
            r"这正是目标参数 $V(\pi_e)$ 的有效影响函数 (EIF) 方程,因此 plug-in",
            r"$\hat V_{TMLE}$ 达到 Cramer-Rao 半参效率界 (van der Laan《Targeted Learning》2011, 定理 5.2)。",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 11)
    return fig


def slide_12_conformal():
    fig, ax = new_slide(
        "U1——Conformal OPE 置信区间 (Taufiq 等,2022)",
        "通过分裂式 conformal 实现免分布、有限样本覆盖",
    )
    math_lines(
        ax,
        [
            r"将数据分成训练集 $\mathcal{D}_{\text{tr}}$ 与校准集 $\mathcal{D}_{\text{cal}}$(大小为 $m$)。",
            r"在 $\mathcal{D}_{\text{tr}}$ 上拟合任意 DR 风格的得分 $\hat\psi$。在校准集上计算残差",
            r"     $R_i \;=\; |\, \hat\psi(z_i) - \mathrm{plug-in}\, |, \quad i \in \mathcal{D}_{\text{cal}}.$",
            "",
            r"取第 $\lceil (1-\alpha)(m+1) \rceil$ 顺序统计量 $\hat q_{1-\alpha}$。",
            r"则 conformal 区间",
            r"     $\widehat C_\alpha \;=\; [\, \hat V - \hat q_{1-\alpha}, \;\; \hat V + \hat q_{1-\alpha}\, ]$",
            r"满足   $\Pr( V \in \widehat C_\alpha ) \,\geq\, 1 - \alpha.$",
            "",
            r"证明梗概。在 $\mathcal{D}_{\text{cal}} \cup \{z_{n+1}\}$ 可交换的前提下,",
            r"$R_{n+1}$ 在 $\{R_1,\dots,R_m,R_{n+1}\}$ 的秩中均匀分布,故",
            r"     $\Pr(R_{n+1} \leq R_{(\lceil(1-\alpha)(m+1)\rceil)}) \;\geq\; 1-\alpha.$",
            "",
            r"主要优点:对 IS 权重的尾部分布无任何参数化假设。",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12.5,
    )
    footer(ax, 12)
    return fig


def slide_13_estimator_validation():
    fig, ax = new_slide(
        "U1——实测验证",
        "200 个随机种子下的合成真值 RMSE 与 CI 覆盖率",
    )
    embed_figure(ax, "fig02_estimator_rmse.png", bbox=(0.04, 0.10, 0.45, 0.70))
    embed_figure(ax, "fig03_ci_coverage_comparison.png", bbox=(0.52, 0.10, 0.45, 0.70))
    ax.text(0.04, 0.07,
            "左图:DM / IPS / DR / DML / TMLE 在合成日志 (n = 5000) 上的 RMSE。"
            "右图:95% CI 覆盖率;即便 IS 权重重尾,conformal 仍可达到名义覆盖。",
            fontsize=11, color=C_GREY, style="italic")
    footer(ax, 13)
    return fig


def slide_14_judge_biases():
    fig, ax = new_slide(
        "U2——单一 LLM 裁判的偏差",
        "三种已被记录的失败模式,会污染奖励信号",
    )
    bullet_list(
        ax,
        [
            "自我偏好偏差——裁判倾向于自家模型族输出。",
            "啰嗦偏差——裁判偏好长文本而非正确答案。",
            "宽松 / 位置偏差——首位答案获胜的概率被显著放大。",
        ],
        y0=0.78, dy=0.055, fontsize=13,
    )
    ax.text(0.06, 0.55, "正式定义(自我偏好):", fontsize=14,
            color=C_TITLE, fontweight="bold")
    math_lines(
        ax,
        [
            r"$B_{\text{self}} \;=\; \mathbb{E}\!\left[\, s_J(a_{\text{self}}) - s_J(a_{\text{other}}) \,|\, q(a_{\text{self}}) = q(a_{\text{other}}) \,\right] \;>\; 0$",
            "",
            r"其中 $s_J$ 是裁判打分,$q$ 是潜在质量。在无偏裁判下,",
            r"$B_{\text{self}}$ 恒为 0。实证上,GPT-4 自评其自身输出 vs. Claude 输出时,",
            r"在相同 prompt 下 $B_{\text{self}} \approx 0.08$ (Zheng 2023, 表 6)。",
        ],
        x=0.06, y0=0.49, dy=0.055, fontsize=13,
    )
    ax.text(0.06, 0.18,
            "对 OPE 的影响:若奖励 y 由裁判定义,$B_{\\text{self}} > 0$ 会使 $\\hat V(\\pi_e)$\n"
            "原样平移同样幅度——偏差直接传递到最终估计上。",
            fontsize=12, color=C_ACCENT)
    footer(ax, 14)
    return fig


def slide_15_triadic_debate():
    fig, ax = new_slide(
        "U2——三方辩论设计",
        "辩护、起诉、裁判——分别来自不同厂商,输出 Beta 后验",
    )
    # diagram on left
    box_w, box_h = 0.20, 0.10
    boxes = [
        (0.06, 0.62, "辩护方(厂商 A)\n论证记录正确"),
        (0.06, 0.46, "起诉方(厂商 B)\n论证记录错误"),
        (0.06, 0.30, "裁判方(厂商 C)\n汇总证据"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#dfe9f3", edgecolor=C_TITLE))
        ax.text(x + box_w / 2, y + box_h / 2, txt, fontsize=11,
                ha="center", va="center", color=C_BODY)
    # arrows
    ax.annotate("", xy=(0.16, 0.40), xytext=(0.16, 0.56),
                arrowprops=dict(arrowstyle="->", color=C_TITLE, lw=1.5))
    ax.annotate("", xy=(0.16, 0.40), xytext=(0.16, 0.30),
                arrowprops=dict(arrowstyle="->", color=C_TITLE, lw=1.5))

    math_lines(
        ax,
        [
            r"输出为正确性 $\theta \in [0,1]$ 上的 Beta 后验:",
            r"     $\theta \,\sim\, \mathrm{Beta}(\alpha,\beta), \quad \alpha = 1 + n_{\text{wins-D}}, \;\; \beta = 1 + n_{\text{wins-P}}.$",
            "",
            r"理论依据(共轭性):把每轮辩论视为以 $\theta$ 为成功率的 Bernoulli 试验,",
            r"以 Jeffreys 先验 $\mathrm{Beta}(1/2, 1/2)$ → 后验 Beta$(\alpha,\beta)$,",
            r"均值 $\alpha/(\alpha+\beta)$,方差 $\alpha\beta\,/\,(\alpha+\beta)^2(\alpha+\beta+1)$。",
            "",
            r"跨厂商多样化对自我偏好偏差的残差给出上界:",
            r"     $|B_{\text{self}}^{\text{triadic}}| \,\leq\, \frac{1}{3}\,\max_v |B_{\text{self}}^v|$",
            r"因为没有任何单一厂商会同时在三个角色中的两个看到自家输出。",
        ],
        x=0.32, y0=0.78, dy=0.055, fontsize=12.5,
    )
    footer(ax, 15)
    return fig


def slide_16_4axis_calibration():
    fig, ax = new_slide(
        "U2——四轴过程奖励 + 校准目标",
        "为什么四个指标缺一不可",
    )
    ax.text(0.06, 0.80, "四个奖励轴:", fontsize=14, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "安全性——是否处理了红旗信号(如自杀风险、脓毒症分诊)。",
            "诊断质量——前 3 鉴别诊断是否命中专家面板。",
            "信息收集——Agent 是否问到了应当问的关键病史。",
            "沟通质量——共情度、术语少、对患者可读。",
        ],
        y0=0.76, dy=0.05, fontsize=12,
    )
    ax.text(0.06, 0.50, "校准目标(对照医学评审员):", fontsize=14,
            color=C_TITLE, fontweight="bold")
    math_lines(
        ax,
        [
            r"$\rho \;\geq\; 0.7$    裁判 vs. 评审员评分的 Pearson 相关",
            r"$\kappa_w \;\geq\; 0.4$    二次加权 Cohen's kappa(序数一致性)",
            r"ICC$(2,1) \;\geq\; 0.75$    类内相关(双因子随机模型,绝对一致性)",
            r"$|\text{bias}| \;\leq\; 0.05$    在 $[0,1]$ 尺度下,裁判减去评审员的均值",
        ],
        x=0.07, y0=0.45, dy=0.055, fontsize=12.5,
    )
    ax.text(0.06, 0.18,
            "为什么是四个?ρ 抓单调差异;κ_w 抓序数错档;\n"
            "ICC 抓 ρ 忽略的尺度漂移;bias 抓 ICC 容许的系统性偏移。\n"
            "任何单一指标都有其他三者会禁止的失败模式。",
            fontsize=11.5, color=C_ACCENT)
    footer(ax, 16)
    return fig


def slide_17_offshelf_fails():
    fig, ax = new_slide(
        "U3——现成嵌入在 OPE 上为何失效",
        "语义相似 ≠ 同一 x 下不同 a 的相似",
    )
    bullet_list(
        ax,
        [
            "MedCPT、BioBERT、SBERT 是为检索 / NLI 训练的——它们按主题聚类。",
            "OPE 需要的嵌入是:对同一名患者写的两份病历应当“接近”,\n   即便其中一份是正确的、另一份是错误的。",
            "现成编码器恰恰相反:患者 i 的错误病历更接近于患者 j 的错误病历\n   (两者都包含同一条幻觉药名),而非更接近 i 的正确病历。",
            "实证:在原始 MedCPT 上,“正确 vs. 反事实”的 AUC = 1.00\n   (编码器只是在按主题区分文档,不是动作)。MIPS 退化。",
        ],
        y0=0.76, dy=0.10, fontsize=13,
    )
    ax.text(0.06, 0.20,
            "CCE 的目标:学一个嵌入 φ(x, a),使得\n"
            "  (1) φ(x, a_correct) 与 φ(x, a_counterfactual) 接近(同一 x);\n"
            "  (2) 当 x ≠ x' 时,φ(x, a) 与 φ(x', a) 远离(动作有判别力);\n"
            "  (3) φ(x, a) 与已观测混杂因子 C 无关。",
            fontsize=12.5, color=C_TITLE)
    footer(ax, 17)
    return fig


def slide_18_cce_objective():
    fig, ax = new_slide(
        "U3——CCE 目标函数(完整推导)",
        "对称 InfoNCE + HSIC 惩罚以保证混杂不变性",
    )
    math_lines(
        ax,
        [
            r"设 $\phi_\theta(x,a)\in\mathbb{R}^d$ 为嵌入网络。对正样本 $(z_i, z_i^+)$,",
            r"用两种数据增强重新编码同一对 $(x_i, a_i^{clin})$;负样本 $\{z_j\}_{j\ne i}$。",
            "",
            r"对称 InfoNCE:",
            r"     $\mathcal{L}_{NCE} \;=\; -\frac{1}{2n}\sum_{i=1}^n [\, \log\frac{e^{\langle z_i, z_i^+\rangle/\tau}}{\sum_j e^{\langle z_i, z_j\rangle/\tau}} \;+\; \log\frac{e^{\langle z_i^+, z_i\rangle/\tau}}{\sum_j e^{\langle z_i^+, z_j\rangle/\tau}} \,]$",
            "",
            r"HSIC 混杂不变性惩罚 (Gretton 2005):",
            r"     $\mathrm{HSIC}(Z, C) \;=\; \frac{1}{(n-1)^2}\,\mathrm{tr}\!(K_Z\, H\, K_C\, H), \quad H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$",
            r"其中 $K_Z, K_C$ 分别为嵌入与混淆变量上的高斯核;",
            r"$\mathrm{HSIC}=0 \;\Leftrightarrow\; Z \,\perp\, C$(普适核定理)。",
            "",
            r"最终目标:",
            r"     $\mathcal{L}(\theta) \;=\; \mathcal{L}_{NCE}(\theta) \;+\; \lambda \cdot \mathrm{HSIC}\!(\phi_\theta(x,a),\, C)$",
            r"$\lambda$ 由验证集选择,使 HSIC ≤ 0.01 同时 $\mathcal{L}_{NCE}$ 与无惩罚最优值差距不超 5%。",
        ],
        x=0.05, y0=0.78, dy=0.052, fontsize=12,
    )
    footer(ax, 18)
    return fig


def slide_19_mips_connection():
    fig, ax = new_slide(
        "U3——与 MIPS 的连接 (Saito & Joachims, 2022)",
        "为何 CCE 是边际化 IPS 所需要的嵌入",
    )
    math_lines(
        ax,
        [
            r"MIPS 估计器:把动作投影到嵌入 $e = \phi(x,a)$,在 $e$ 空间里加权:",
            r"     $\hat V_{MIPS} \;=\; \frac{1}{n}\sum_i \frac{\pi_e(e_i\mid x_i)}{\pi_b(e_i\mid x_i)}\, y_i.$",
            "",
            r"定理 (Saito & Joachims 2022, 定理 3.1, 改述)。",
            r"若“无直接效应”条件成立:$Y \,\perp\!\!\!\perp\, A \,\mid\, X, \phi(X,A)$,",
            r"则 $\hat V_{MIPS}$ 对 $V(\pi_e)$ 无偏,且",
            r"     $\mathrm{Var}(\hat V_{MIPS}) \;\leq\; \mathrm{Var}(\hat V_{IPS})$",
            r"当 $\phi$ 在 $\mathcal{A}$ 上不可逆时严格不等。",
            "",
            r"CCE 提供定理所要求的嵌入:",
            r"     · InfoNCE 使 $\phi$ 成为动作引发结局变化的充分统计量,",
            r"     · HSIC 强制 $\phi \,\perp\!\!\!\perp\, C$,移除残余的混淆通道。",
            "",
            r"净结果:在合成基准上方差降低 5-50 倍(见第 13 页)。",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    footer(ax, 19)
    return fig


def slide_20_cce_headline():
    fig, ax = new_slide(
        "U3——真实 ACI-Bench 上的核心结果",
        "原始 MedCPT AUC = 1.00(退化)→ CCE 后 AUC = 0.55(可识别)",
    )
    embed_figure(ax, "fig04_cce_real_headline.png", bbox=(0.06, 0.10, 0.65, 0.72))
    ax.text(0.74, 0.74, "图意说明", fontsize=14, color=C_TITLE,
            fontweight="bold")
    bullet_list(
        ax,
        [
            "柱形:正确 vs. 反事实\n   分类器的 AUC。",
            "原始 MedCPT:AUC = 1.00\n   (编码器按主题分,\n   OPE 退化)。",
            "CCE 训练后:AUC = 0.55,\n   接近可识别区间。",
            "首次在真实医学文本\n   (ACI-Bench)上证明。",
        ],
        x=0.74, y0=0.66, dy=0.10, fontsize=10.5, color=C_BODY,
    )
    footer(ax, 20)
    return fig


def slide_21_cce_caveat():
    fig, ax = new_slide(
        "U3——诚实说明",
        "哪些是真实数据、哪些是模板生成、哪些尚待完成",
    )
    bullet_list(
        ax,
        [
            "真实:ACI-Bench 转录文本(医生病历)、MedCPT 基础编码器、\n   CCE 训练运行、真实留出集上 AUC = 0.55 的结果。",
            "模板生成:目前“Agent 反事实”是确定性扰动\n   (替换药名、剂量、删除红旗信号)——并非真正的 LLM 重跑。",
            "待完成:在约 500 个病例上做真实 LLM Agent 重跑(预算 300 美元 / 2 天)。\n   预期定性结论(AUC 下降)保留,而每例动作分布更接近真实。",
            "为何现在发表?方法论贡献(CCE 打破退化嵌入)与反事实如何生成无关;\n   LLM 重跑只是稳健性检查,而非主要论据。",
        ],
        y0=0.76, dy=0.13, fontsize=12.5,
    )
    footer(ax, 21)
    return fig


def slide_22_pse_decomposition():
    fig, ax = new_slide(
        "U4——路径特异效应分解",
        "在 4 个奖励轴 × 各专科分层上分解 ΔV",
    )
    math_lines(
        ax,
        [
            r"从医生策略 $\pi_b$ 切换到 Agent $\pi_e$ 的总效应:",
            r"     $\Delta \;=\; V(\pi_e) - V(\pi_b) \;=\; \sum_{k=1}^{4}\, \sum_{s \in \mathcal{S}}\, \Delta_{k,s}$",
            r"     其中 $k$ 索引奖励轴(安全 / 诊断 / 信息 / 沟通),$s$ 索引专科分层",
            r"     (心内、皮肤、神经……)。",
            "",
            r"逐分量效应,采用中介公式形式 (Pearl 2001):",
            r"     $\Delta_{k,s} \;=\; \mathbb{E}_{x\in s}\![\,\mathbb{E}_{a\sim\pi_e}\, Y_k(x,a) \;-\; \mathbb{E}_{a\sim\pi_b}\, Y_k(x,a)\,]$",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    embed_figure(ax, "fig07_4axis_decomposition.png", bbox=(0.06, 0.08, 0.88, 0.32))
    footer(ax, 22)
    return fig


def slide_23_msm():
    fig, ax = new_slide(
        "U4——Logistic 边际敏感度模型 (MSM)",
        "Yadlowsky 2018:用 Γ 限制未观测混杂带来的偏倚",
    )
    math_lines(
        ax,
        [
            r"假设 (Tan 2006)。未观测混淆变量 $U$ 至多让处理对数几率漂移 $\log\Gamma$:",
            r"     $\frac{1}{\Gamma} \,\leq\, \frac{\pi_b(a\mid x, u)\,/\,(1-\pi_b(a\mid x, u))}{\pi_b(a\mid x)\,/\,(1-\pi_b(a\mid x))} \,\leq\, \Gamma.$",
            "",
            r"Yadlowsky (2018) 第 4.1 节:在该 MSM 下,平均处理效应的最坏情形",
            r"下界为",
            r"     $\Delta_{lo}(\Gamma) \;=\; \Delta \;-\; \frac{\Gamma - 1}{\Gamma + 1}\, \mathbb{E}|d_i|, \qquad d_i \,=\, w_i\,(y_i - \hat f(x_i, a_i)).$",
            "",
            r"推导梗概。MSM 推出 $|\hat w_i^{adv} - \hat w_i| \,\leq\, (\Gamma-1)/(\Gamma+1)\,\hat w_i$",
            r"对所有与 $\Gamma$ 相容的对抗性重加权成立。对 DR 得分 $d_i$ 应用 Cauchy-Schwarz 即得",
            r"上式所示的单边下界。$\Delta_{hi}(\Gamma)$ 同理对称。",
            "",
            r"单调性:$\partial\Delta_{lo}/\partial\Gamma \,=\, -2/(\Gamma+1)^2 \cdot \mathbb{E}|d_i| \,<\, 0$,",
            r"随设定的混杂强度增大,边界单调展宽。",
        ],
        x=0.06, y0=0.78, dy=0.054, fontsize=12,
    )
    footer(ax, 23)
    return fig


def slide_24_fragility():
    fig, ax = new_slide(
        "U4——脆弱度 Γ*:闭式解",
        "刚好令核心结果失效所需的最小未观测混杂强度",
    )
    math_lines(
        ax,
        [
            r"将 $\Gamma^*$ 定义为 $\Delta_{lo}(\Gamma) = 0$ 处的最小 $\Gamma$:",
            r"     $\Delta - \frac{\Gamma^* - 1}{\Gamma^* + 1}\, \mathbb{E}|d_i| \;=\; 0$",
            "",
            r"令 $r = \Delta\,/\,\mathbb{E}|d_i|$(“效应—噪声比”)。求解:",
            r"     $r(\Gamma^* + 1) \;=\; \Gamma^* - 1 \;\;\Leftrightarrow\;\; \Gamma^* \;=\; \frac{1 + r}{1 - r}.$",
            "",
            r"解读:$\Gamma^* \approx 1.2$ 即脆弱(任何 OR > 1.2 的未观测混淆变量都能解释掉结果);",
            r"按惯例,$\Gamma^* > 2$ 被视为鲁棒。",
        ],
        x=0.06, y0=0.78, dy=0.055, fontsize=12.5,
    )
    embed_figure(ax, "fig06_msm_sensitivity_curve.png", bbox=(0.06, 0.06, 0.88, 0.32))
    footer(ax, 24)
    return fig


def slide_25_workflow():
    fig, ax = new_slide(
        "工程实现——10 步流水线",
        "每一步对应四项升级中的某一项",
    )
    embed_figure(ax, "fig08_workflow_summary.png", bbox=(0.04, 0.10, 0.62, 0.72))
    bullet_list(
        ax,
        [
            "01 问题定义 (U0)",
            "02 数据审计 (U0)",
            "03 主动采样 (U2)",
            "04 运行裁判 (U2)",
            "05 运行 Agent (U2/U3)",
            "06 训练 CCE (U3)",
            "07 构建真值 (U2)",
            "08 合成基准 (U1)",
            "09 主实验 (U1+U3)",
            "10 失败模式 (U4)",
        ],
        x=0.69, y0=0.76, dy=0.055, fontsize=12,
    )
    footer(ax, 25)
    return fig


def slide_26_layer1():
    fig, ax = new_slide(
        "工程实现——真实 ACI-Bench 第一层审计",
        "在原始转录上做语境感知的质量检测",
    )
    embed_figure(ax, "fig05_layer1_audit_real.png", bbox=(0.04, 0.10, 0.60, 0.72))
    bullet_list(
        ax,
        [
            "第一层 = 转录级 QC。",
            "可检测:空轮、角色错位、\n   PII 泄漏、跑题闲聊。",
            "语境感知:同一词\n   (如“胸”)在心内科正常,\n   在皮肤科则是红旗。",
            "真实 ACI-Bench 通过率:\n   约 93%;被标记的样本\n   送入人工复核。",
        ],
        x=0.66, y0=0.76, dy=0.105, fontsize=11.5,
    )
    footer(ax, 26)
    return fig


def slide_27_repo_stats():
    fig, ax = new_slide(
        "工程实现——仓库统计",
        "可复现性与 CI 表面",
    )
    stats = [
        ("提交数", "15"),
        ("通过测试数", "161"),
        ("Python 文件数", "75+"),
        ("端到端 mock 模式", "支持(无需 API key)"),
        ("GitHub Actions CI", "推送时执行 lint + 测试"),
        ("Docker 镜像", "ccema:latest, ~2.1 GB"),
        ("依赖锁文件", "requirements-lock.txt(版本锁定)"),
        ("LICENSE", "MIT"),
    ]
    y = 0.74
    for k, v in stats:
        ax.text(0.10, y, k, fontsize=14, color=C_TITLE, fontweight="bold")
        ax.text(0.42, y, v, fontsize=14, color=C_BODY)
        y -= 0.07
    ax.text(0.06, 0.10,
            "Mock 模式意味着:scripts/01_*.py 至 scripts/10_*.py 的每个脚本都能端到端运行,\n"
            "不依赖任何外部 API;评审者可以在 10 分钟内复现核心数字。",
            fontsize=11.5, color=C_ACCENT)
    footer(ax, 27)
    return fig


def slide_28_status():
    fig, ax = new_slide(
        "状态——真实数据 vs. 模拟验证",
        "当前数字到底建立在什么基础上",
    )
    rows = [
        ("U1(现代 OPE)", "MOCK", "已知 V 的合成日志;待补真实 LLM 输出"),
        ("U2(辩论裁判)", "MOCK", "厂商 API 已 stub;待补医学评审员校准集"),
        ("U3(CCE)",     "REAL", "在真实 ACI-Bench 上训练并评估(AUC 1.00 → 0.55)"),
        ("U4(分解 + MSM)", "MOCK", "脆弱度闭式已被解析与合成数据共同验证"),
    ]
    y_head = 0.78
    ax.add_patch(Rectangle((0.04, y_head - 0.06), 0.92, 0.06,
                           facecolor=C_TITLE, edgecolor="none"))
    ax.text(0.07, y_head - 0.03, "升级项", color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(0.30, y_head - 0.03, "状态", color="white", fontsize=13, fontweight="bold", va="center")
    ax.text(0.45, y_head - 0.03, "说明", color="white", fontsize=13, fontweight="bold", va="center")
    y = y_head - 0.12
    for name, state, notes in rows:
        col = "#27ae60" if state == "REAL" else "#e67e22"
        ax.text(0.07, y, name, fontsize=13, color=C_BODY, va="center")
        ax.text(0.30, y, state, fontsize=13, color=col, fontweight="bold", va="center")
        ax.text(0.45, y, notes, fontsize=11.5, color=C_BODY, va="center")
        y -= 0.09
    ax.text(0.06, 0.18,
            "核心科学结论(CCE 在真实医学文本上打破退化嵌入)\n"
            "完全建立在真实数据上。工程性结论建立在 mock + 计划中的真实重跑之上。",
            fontsize=12, color=C_ACCENT, fontweight="bold")
    footer(ax, 28)
    return fig


def slide_29_roadmap():
    fig, ax = new_slide(
        "下一步路线图——未来 5 周",
        "从今天的 v1 走到可投稿的 v2",
    )
    weeks = [
        ("第 1 周", "在约 500 例 ACI-Bench 上跑真实 LLM Agent",
         "预算 300 美元,约 2 天计算。替代 U3 中的模板反事实。"),
        ("第 2 周", "金标签真值标注",
         "招募 2-3 名持证医师评审员;200 例 × 4 个轴。"),
        ("第 3-4 周", "校准集 + 裁判调优",
         "在 holdout 上达到 ρ ≥ 0.7、κ_w ≥ 0.4、ICC ≥ 0.75、|bias| ≤ 0.05。"),
        ("第 5 周", "U1 + U4 在真实数据上的运行",
         "对真实 V(π_e) - V(π_b) 跑 DML / TMLE + MSM Γ*。"),
        ("第 6-7 周", "论文撰写 + 消融实验",
         "目标 NeurIPS 2026 或 JAMIA 短文。"),
    ]
    y = 0.78
    for w, title, desc in weeks:
        ax.add_patch(FancyBboxPatch((0.05, y - 0.10), 0.10, 0.085,
                                    boxstyle="round,pad=0.005",
                                    facecolor=C_TITLE, edgecolor="none"))
        ax.text(0.10, y - 0.058, w, fontsize=13, color="white",
                fontweight="bold", ha="center", va="center")
        ax.text(0.18, y - 0.02, title, fontsize=13, color=C_TITLE, fontweight="bold")
        ax.text(0.18, y - 0.07, desc, fontsize=11.5, color=C_BODY)
        y -= 0.13
    footer(ax, 29)
    return fig


def slide_30_risks():
    fig, ax = new_slide(
        "风险与对策",
        "可能打破计划的三件事,以及我们的应对",
    )
    risks = [
        ("无法及时获得医学评审员",
         "用 HealthBench (Anthropic 2025) 作为锚点:在其量表上重打分,\n汇报相对排序。失去绝对校准,但保留相对比较结论。"),
        ("MSM 边界扩张过快(Γ* < 1.2)",
         "把核心结果改述为“脆弱发现”——仍可作为方法论演示发表;\n贡献本身在于 Γ* 的闭式解。"),
        ("CCE 在真实 LLM 重跑上未能压低 AUC",
         "说明模板反事实比真实容易(信息量丰富的负结果)。\n论文中加入仅 HSIC 的消融位置。"),
        ("厂商 API 在实验中途修改打分逻辑",
         "锁定模型版本(gpt-4-1106、claude-3.5-sonnet-20240620、gemini-1.5-pro-002),\n保存原始裁判转录,任何漂移可事后追查。"),
    ]
    y = 0.78
    for risk, mit in risks:
        ax.text(0.06, y, "风险:" + risk, fontsize=12.5, color=C_ACCENT, fontweight="bold")
        ax.text(0.10, y - 0.04, "对策:" + mit, fontsize=11, color=C_BODY)
        y -= 0.16
    footer(ax, 30)
    return fig


def slide_31_takehome():
    fig, ax = new_slide(
        "核心信息",
        "一句话主张,后接三句论据",
    )
    ax.text(0.06, 0.74,
            "在真实医学嵌入上,U3 拯救了 IPS 类估计器。",
            fontsize=22, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "原始 MedCPT 区分正确与反事实病历的 AUC = 1.00\n   ——OPE 问题完全退化,IS 权重要么 0 要么 ∞。",
            "CCE 训练后 AUC 降到 0.55——进入现代 OPE 估计器\n   (DML、TMLE、MIPS)具有有限方差解的可识别区间。",
            "这是首次在真实 ACI-Bench 转录文本上的演示;\n   就方法论主张而言,无需真实 LLM 重跑(模板反事实即可成立)。",
        ],
        y0=0.58, dy=0.13, fontsize=13,
    )
    ax.text(0.06, 0.13,
            "若以上结论在第 1 周的 LLM 重跑后仍成立,U3 就是核心科学贡献,\n"
            "U1、U2、U4 则是围绕它的可复现工程支架。",
            fontsize=12, color=C_ACCENT)
    footer(ax, 31)
    return fig


def slide_32_thanks():
    fig, ax = new_slide(
        "致谢与联系方式",
    )
    ax.text(0.06, 0.74, "感谢:", fontsize=18, color=C_TITLE, fontweight="bold")
    bullet_list(
        ax,
        [
            "ACI-Bench 作者团队提供公开的临床对话语料。",
            "MedCPT 团队 (NIH) 提供开源医学文本编码器。",
            "Anthropic / Claude Code 提供智能体化的构建工具链。",
            "对 OPE 假设进行了压力测试的评审者与同事。",
        ],
        y0=0.66, dy=0.06, fontsize=13,
    )
    ax.text(0.06, 0.36, "联系方式与代码仓库:", fontsize=18, color=C_TITLE, fontweight="bold")
    ax.text(0.08, 0.30, "郝林浩 Linhao Hao   <lhhao0430@gmail.com>", fontsize=14, color=C_BODY)
    ax.text(0.08, 0.25,
            "github.com/<user>/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents",
            fontsize=13, color=C_BODY)
    ax.text(0.08, 0.20, "日期:2026-05-04   |   版本:v1   |   许可证:MIT",
            fontsize=12, color=C_GREY)
    ax.text(0.5, 0.08, "谢谢!", fontsize=22, color=C_ACCENT,
            fontweight="bold", ha="center")
    footer(ax, 32)
    return fig


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
SLIDE_FUNCS = [
    slide_01_title,
    slide_02_eval_gap,
    slide_03_ope_tte_gap,
    slide_04_setup,
    slide_05_estimand,
    slide_06_assumptions,
    slide_07_upgrades_table,
    slide_08_dm_ips_dr,
    slide_09_dr_overfitting_bias,
    slide_10_dml,
    slide_11_tmle,
    slide_12_conformal,
    slide_13_estimator_validation,
    slide_14_judge_biases,
    slide_15_triadic_debate,
    slide_16_4axis_calibration,
    slide_17_offshelf_fails,
    slide_18_cce_objective,
    slide_19_mips_connection,
    slide_20_cce_headline,
    slide_21_cce_caveat,
    slide_22_pse_decomposition,
    slide_23_msm,
    slide_24_fragility,
    slide_25_workflow,
    slide_26_layer1,
    slide_27_repo_stats,
    slide_28_status,
    slide_29_roadmap,
    slide_30_risks,
    slide_31_takehome,
    slide_32_thanks,
]


def build():
    print(f"Building {len(SLIDE_FUNCS)} slides (zh)...")
    figs = []
    for i, fn in enumerate(SLIDE_FUNCS, start=1):
        try:
            fig = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! slide {i} ({fn.__name__}) failed: {exc}")
            raise
        figs.append(fig)
        print(f"  - slide {i:02d} {fn.__name__}")

    # ---- PDF
    print(f"Writing PDF -> {PDF_PATH}")
    with PdfPages(PDF_PATH) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches=None, pad_inches=0,
                        facecolor=fig.get_facecolor())

    # ---- PPTX
    print(f"Writing PPTX -> {PPTX_PATH}")
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]  # blank
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150,
                    facecolor=fig.get_facecolor(),
                    bbox_inches=None, pad_inches=0)
        buf.seek(0)
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(buf, 0, 0,
                                 width=prs.slide_width,
                                 height=prs.slide_height)
        plt.close(fig)
    prs.save(str(PPTX_PATH))

    # ---- Report
    pdf_size = PDF_PATH.stat().st_size
    pptx_size = PPTX_PATH.stat().st_size
    print()
    print("=" * 60)
    print(f"Slides:           {len(SLIDE_FUNCS)}")
    print(f"PDF:              {PDF_PATH}  ({pdf_size/1024:.1f} KB)")
    print(f"PPTX:             {PPTX_PATH}  ({pptx_size/1024:.1f} KB)")
    print(f"Total size:       {(pdf_size + pptx_size)/1024:.1f} KB")
    if PLACEHOLDERS_USED:
        print(f"Placeholder figs: {len(PLACEHOLDERS_USED)}")
        for f in PLACEHOLDERS_USED:
            print(f"  - {f}")
    else:
        print("Placeholder figs: 0")
    print("=" * 60)


if __name__ == "__main__":
    build()
