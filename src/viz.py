"""시각화 — Seaborn 정적 차트(PNG), Plotly 인터랙티브 차트(HTML)"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

matplotlib.use("Agg")  # 파일 저장 전용 백엔드
# 기호/한글 폴백 — macOS 우선, 없으면 Windows(Malgun Gothic)/Linux(Noto Sans CJK KR) 순
plt.rcParams["font.family"] = [
    "Helvetica Neue",
    "AppleGothic",
    "Malgun Gothic",
    "Noto Sans CJK KR",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 색 역할 고정 — 계열색은 순서대로만 배정하고 순환시키지 않는다
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SUB = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOW, HIGH = "#2a78d6", "#eb6834"  # <=50K / >50K
# 상관행렬은 부호가 있는 값이므로 발산형(파랑↔빨강, 중앙은 무채색)
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr",
    [
        "#104281",
        "#256abf",
        "#3987e5",
        "#9ec5f4",
        "#f0efec",
        "#f6c9c8",
        "#ee8b8a",
        "#e34948",
        "#b52c2b",
        "#8a1e1d",
    ],
)

NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
POSITIVE = ">50K"


def _style(ax, title, grid_axis="y"):
    # 격자·축을 배경 한 단계 위 헤어라인으로 낮추고 제목/눈금 색을 통일
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    ax.set_facecolor(SURFACE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_SUB)
    ax.yaxis.label.set_color(INK_SUB)


def _high_ratio(df, col):
    # 지정 컬럼별 고소득 비율과 표본 수
    return (
        df.assign(high=df["income"].eq(POSITIVE)).groupby(col)["high"].agg(ratio="mean", n="count")
    )


def save_seaborn_charts(df, file_path):
    # 소득 격차를 네 각도(연령·근로시간·직업·수치형 상관)에서 보여주는 PNG 저장
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=SURFACE)
    order = ["<=50K", POSITIVE]
    palette = {"<=50K": LOW, POSITIVE: HIGH}

    # (1) 연령 분포 — 두 그룹의 표본 수가 크게 다르므로 밀도로 정규화해 비교
    ax = axes[0][0]
    sns.histplot(
        data=df,
        x="age",
        hue="income",
        hue_order=order,
        palette=palette,
        bins=36,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.30,
        linewidth=2,
        ax=ax,
    )
    _style(ax, "연령 분포 — 소득 그룹별 (밀도 정규화)")
    ax.set_xlabel("나이"), ax.set_ylabel("밀도")
    legend = ax.get_legend()
    legend.set_title(""), legend.get_frame().set_visible(False)

    # (2) 주당 근로시간 — t-test 결과를 눈으로 확인하는 분포 비교
    ax = axes[0][1]
    sns.boxplot(
        data=df,
        x="hours-per-week",
        y="income",
        order=order,
        hue="income",
        hue_order=order,
        palette=palette,
        legend=False,
        width=0.42,
        showfliers=False,
        linewidth=1.2,
        ax=ax,
    )
    _style(ax, "주당 근로시간 — 소득 그룹별 (상자=IQR, 수염=1.5×IQR)", grid_axis="x")
    ax.set_xlabel("주당 근로시간"), ax.set_ylabel("")
    ax.set_xlim(0, 72), ax.set_ylim(1.45, -0.45)  # 0시간 기준 + 상자 여백 확보
    for i, grp in enumerate(order):  # 평균만 선택적으로 직접 라벨
        mean = df.loc[df["income"] == grp, "hours-per-week"].mean()
        ax.scatter(mean, i, s=46, color=SURFACE, zorder=3, edgecolor=palette[grp], linewidth=2)
        ax.text(mean, i - 0.22, f"평균 {mean:.1f}h", ha="center", fontsize=9, color=INK_SUB)

    # (3) 직업별 고소득 비율 — 순위가 메시지이므로 값 순 정렬 + 값 직접 라벨
    ax = axes[1][0]
    occ = _high_ratio(df, "occupation").sort_values("ratio")
    ax.barh(occ.index, occ["ratio"], height=0.72, color=LOW)
    overall = df["income"].eq(POSITIVE).mean()
    ax.axvline(overall, color=INK_MUTED, linewidth=1)
    ax.text(
        overall + 0.006,
        1.005,
        f"전체 {overall:.1%}",
        fontsize=9,
        color=INK_SUB,
        transform=ax.get_xaxis_transform(),
    )
    _style(ax, "직업별 고소득(>50K) 비율", grid_axis="x")
    ax.set_xlabel(">50K 비율"), ax.set_xlim(0, occ["ratio"].max() * 1.18)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for y, v in enumerate(occ["ratio"]):
        ax.text(v + 0.008, y, f"{v:.1%}", va="center", fontsize=9, color=INK_SUB)

    # (4) 수치형 상관 — 대각·상삼각은 정보가 없어 가림
    ax = axes[1][1]
    corr = df[NUM_COLS].corr().iloc[1:, :-1]  # 대각·빈 행열 제거
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    hm = sns.heatmap(
        corr,
        mask=mask,
        cmap=CORR_CMAP,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=2,
        linecolor=SURFACE,
        cbar_kws={"shrink": 0.6, "pad": 0.03},
        ax=ax,
    )
    hm.collections[0].colorbar.ax.set_title("상관계수", fontsize=9, color=INK_SUB, pad=8)
    for t in ax.texts:
        t.set_color("#ffffff" if abs(float(t.get_text())) > 0.55 else INK)
    _style(ax, "수치형 변수 상관계수 (하삼각)", grid_axis=None)
    ax.tick_params(labelrotation=0), ax.set_xticklabels(
        ax.get_xticklabels(), rotation=20, ha="right"
    )

    fig.tight_layout(pad=2.4)
    try:
        fig.savefig(file_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    except OSError as e:
        raise SystemExit(f"[오류] 정적 차트 저장 실패: {e}") from e
    finally:
        plt.close(fig)


def save_plotly_chart(df, file_path):
    # 학력별 고소득 비율을 학력 순서대로 세운 인터랙티브 막대 차트로 HTML 저장
    ratio = _high_ratio(df, "education")
    order = df.groupby("education")["education-num"].first().sort_values()
    ratio = ratio.loc[order.index]  # 학력 연차 순 = 추세가 보이는 순서
    overall = df["income"].eq(POSITIVE).mean()

    fig = go.Figure(
        go.Bar(
            x=ratio["ratio"],
            y=ratio.index,
            orientation="h",
            marker_color=LOW,
            marker_line_width=0,
            customdata=ratio["n"],
            text=[f"{v:.1%}" for v in ratio["ratio"]],
            textposition="outside",
            textfont={"size": 11, "color": INK_SUB},
            hovertemplate="<b>%{y}</b><br>>50K 비율 %{x:.1%}<br>표본 %{customdata:,}명<extra></extra>",
        )
    )
    fig.add_vline(
        x=overall,
        line_color=INK_MUTED,
        line_width=1,
        annotation_text=f"전체 {overall:.1%}",
        annotation_position="top",
        annotation_font={"size": 11, "color": INK_SUB},
    )
    fig.update_layout(
        title={
            "text": "학력별 고소득(>50K) 비율<br>"
            f"<span style='font-size:12px;color:{INK_MUTED}'>"
            f"학력 연차 낮은 순 · 정제 후 {len(df):,}명</span>",
            "font": {"size": 18, "color": INK},
            "x": 0.02,
        },
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_SUB},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        bargap=0.34,
        height=680,
        margin={"l": 150, "r": 60, "t": 96, "b": 56},
        showlegend=False,
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": AXIS, "font_size": 12},
    )
    fig.update_xaxes(
        title=">50K 비율",
        tickformat=".0%",
        gridcolor=GRID,
        zeroline=False,
        linecolor=AXIS,
        ticks="",
        range=[0, ratio["ratio"].max() * 1.15],
    )
    fig.update_yaxes(title="", showgrid=False, linecolor=AXIS, ticks="")
    try:
        fig.write_html(file_path)  # plotly.js 인라인 — 오프라인에서도 열림
    except OSError as e:
        raise SystemExit(f"[오류] 인터랙티브 차트 저장 실패: {e}") from e


# =========================================================
# 전체 변수 개별 플롯 (부록) — 수치형 6개 / 범주형 8개
# =========================================================
NUM_ALL = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
# 범주 수가 비슷한 변수끼리 같은 행에 두고 행 높이를 범주 수에 비례시킨다
CAT_LAYOUT = [
    ("education", "occupation"),
    ("native-country", "workclass"),
    ("marital-status", "relationship"),
    ("race", "sex"),
]
FOLD_MIN, TOP_N = 20, 14  # 범주가 20개를 넘을 때만(native-country) 상위 14개 + 기타


def _income_legend(fig, order, palette):
    # 패널마다 반복되는 범례를 그림 단위로 한 번만 둔다
    handles = [
        Patch(facecolor=palette[g], alpha=0.30, edgecolor=palette[g], linewidth=2, label=g)
        for g in order
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=11,
        labelcolor=INK_SUB,
        bbox_to_anchor=(0.995, 0.995),
    )


def save_numeric_panels(df, file_path):
    # 수치형 6개 변수의 소득 그룹별 분포를 한 장에 저장
    order = ["<=50K", POSITIVE]
    palette = {"<=50K": LOW, POSITIVE: HIGH}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor=SURFACE)

    for ax, col in zip(axes.ravel(), NUM_ALL, strict=True):
        zero_share = (df[col] == 0).mean()
        skewed = bool(zero_share > 0.5)  # capital-gain/loss는 대부분 0
        data = df[df[col] > 0] if skewed else df
        discrete = data[col].nunique() <= 20  # education-num처럼 정수 눈금
        hi = data[col].quantile(0.995)
        clipped = not skewed and not discrete and data[col].max() > hi * 1.5

        sns.histplot(
            data=data,
            x=col,
            hue="income",
            hue_order=order,
            palette=palette,
            bins=36,
            discrete=discrete,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.30,
            linewidth=2,
            log_scale=(skewed, False),
            ax=ax,
        )
        note = (
            f"  (0이 아닌 {1 - zero_share:.1%}만 · 로그 축)"
            if skewed
            else "  (상위 0.5% 절단)" if clipped else ""
        )
        _style(ax, f"{col}{note}")
        ax.set_xlabel(""), ax.set_ylabel("밀도")
        if clipped:
            ax.set_xlim(right=hi)
        if skewed:
            ax.xaxis.set_minor_formatter(NullFormatter())  # 로그축 보조 눈금 라벨 정리
        if ax.get_legend():
            ax.get_legend().remove()

    _income_legend(fig, order, palette)
    fig.suptitle(
        "수치형 변수 분포 — 소득 그룹별 (밀도 정규화)",
        fontsize=14,
        color=INK,
        x=0.006,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(pad=1.4, rect=(0, 0, 1, 0.965))
    try:
        fig.savefig(file_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    except OSError as e:
        raise SystemExit(f"[오류] 수치형 분포 차트 저장 실패: {e}") from e
    finally:
        plt.close(fig)


def _fold_rare(ratio, top_n=TOP_N):
    # 표본 수 상위 top_n개만 남기고 나머지는 가중평균 한 묶음으로
    if len(ratio) <= FOLD_MIN:
        return ratio
    keep = ratio.nlargest(top_n, "n")
    rest = ratio.drop(keep.index)
    folded = {"ratio": (rest["ratio"] * rest["n"]).sum() / rest["n"].sum(), "n": rest["n"].sum()}
    keep.loc[f"기타 {len(rest)}개"] = folded
    return keep


def save_categorical_panels(df, file_path):
    # 범주형 8개 변수별 고소득 비율을 한 장에 저장 (값 순 정렬 + 전체 비율 기준선)
    ratios = {
        c: _fold_rare(_high_ratio(df, c)).sort_values("ratio") for row in CAT_LAYOUT for c in row
    }
    heights = [max(len(ratios[a]), len(ratios[b])) for a, b in CAT_LAYOUT]
    overall = df["income"].eq(POSITIVE).mean()

    fig, axes = plt.subplots(
        len(CAT_LAYOUT),
        2,
        figsize=(17, sum(heights) * 0.40 + 3),
        height_ratios=heights,
        facecolor=SURFACE,
    )
    for (left, right), row_axes, rows in zip(CAT_LAYOUT, axes, heights, strict=True):
        for col, ax in zip((left, right), row_axes, strict=True):
            r = ratios[col]
            labels = [f"{i} ({n:,})" for i, n in zip(r.index, r["n"], strict=True)]
            ax.barh(labels, r["ratio"], height=0.72, color=LOW)
            ax.axvline(overall, color=INK_MUTED, linewidth=1)
            _style(ax, f"{col} — 고소득(>50K) 비율", grid_axis="x")
            ax.set_xlim(0, max(r["ratio"].max() * 1.20, overall * 1.3))
            pad = (rows - len(r)) / 2  # 행 안에서 막대 두께 통일 + 수직 중앙
            ax.set_ylim(-0.6 - pad, len(r) - 0.4 + pad)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            for y, v in enumerate(r["ratio"]):
                ax.text(v + 0.006, y, f"{v:.1%}", va="center", fontsize=9, color=INK_SUB)

    fig.suptitle(
        f"범주형 변수별 고소득(>50K) 비율  ·  괄호는 표본 수  ·  세로선 = 전체 {overall:.1%}",
        fontsize=14,
        color=INK,
        x=0.008,
        ha="left",
        y=0.997,
    )
    fig.tight_layout(pad=2.0, rect=(0, 0, 1, 0.985))
    try:
        fig.savefig(file_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    except OSError as e:
        raise SystemExit(f"[오류] 범주형 비율 차트 저장 실패: {e}") from e
    finally:
        plt.close(fig)
