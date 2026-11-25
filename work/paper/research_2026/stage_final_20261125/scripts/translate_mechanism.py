"""Translate the existing editable figure while keeping its native geometry and math runs."""
import re
from pathlib import Path

from pptx import Presentation


FIGURES = Path(__file__).resolve().parents[1] / "source" / "figures"
LABELS = {
    "训练与推理共用": "Train and inference",
    "  前锚": "  earlier",
    "  后锚": "  later",
    "两期精细参考图块": "Two fine anchors",
    "共享编码器 E": "Shared E",
    "Softmax 归一化": "Softmax",
    "同网格的三期粗观测": "Three coarse observations",
    "同权重编码器 E": "Shared E",
    "精细成分 · 双锚插值": "Fine anchor interpolation",
    "日期": "Time",
    "加权": "blend",
    "、w": ", w",
    " 由日期间隔确定": " follow day gaps",
    "粗成分 · 目标日时间偏差": "Coarse temporal deviation",
    "参考": "Ref.",
    "插值": "blend",
    "目标日实际权重 − 粗参考插值": "Target minus coarse anchor blend",
    "目标日成分比例传递": "Target component weights",
    "截断到下界": "Floor",
    "按和归一化": "Normalize",
    "  8 通道目标权重": "  8 target weights",
    "1×1 条件投影": "1×1 projection",
    "8 通道": "8 ch.",
    "卷积": "PW",
    "64 通道": "64 ch.",
    "固定缩放 ×0.05": "Scale ×0.05",
    "单步重建路径": "One-step fusion",
    "原始观测 O": "Inputs O",
    "精细卷积": "Fine conv.",
    "粗图卷积": "C conv.",
    "拼接投影": "Concat + conv.",
    "  64 通道": "  64 ch.",
    "成分条件增量  Δh": "Condition update  Δh",
    "推理  纯高斯": "Test  Gaussian",
    "训练  真值加噪": "Train  noisy GT",
    "噪声卷积": "Noise conv.",
    "双流 STFDiff · 一次主干调用": "Dual-stream STFDiff · one call",
    "各尺度  N": "Scale  N",
    "条件流": "Cond.",
    "噪声流": "Noise",
    "  编码": "  enc.",
    "C 中间块": "C mid",
    "N 中间块": "N mid",
    "跳接": "Skip",
    "解码": "decode",
    "目标日精细重建": "Target fine image",
    "6 波段": "6 bands",
    "裁剪至 [−1,1]": "Clip to [−1,1]",
    "可学习": "Learned",
    "固定规则": "Fixed",
    "仅训练": "Train",
    "训练监督": "Train loss",
    "双锚光谱重建 ×0.05": "Anchor spectral loss ×0.05",
    ") 重建反射率": ") reconstructs spectra",
    "目标真值 F": "Target GT F",
    "目标图像 L1  L": "Image L1  L",
}


def main():
    figure = Presentation(str(FIGURES / "mechanism_zh.pptx"))
    for slide in figure.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = LABELS.get(run.text, run.text)
                    if re.search(r"[\u3400-\u9fff]", run.text):
                        raise ValueError(f"Untranslated {shape.name}: {run.text}")
    figure.save(str(FIGURES / "mechanism_en.pptx"))


if __name__ == "__main__":
    main()
