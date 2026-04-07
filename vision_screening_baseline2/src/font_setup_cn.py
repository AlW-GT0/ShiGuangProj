# font_setup_cn.py
# 作用：
# 1) 自动下载可用中文字体到本地目录
# 2) 注册到 matplotlib
# 3) 设置全局字体 rcParams，解决中文方块问题

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


NOTO_SANS_SC_URLS = [
    # Noto CJK 官方仓库（GitHub）中的简体中文 Sans Regular
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
    # 备用：如果上面路径未来变动，可自行替换为其它字体链接
]


def download_file(url: str, dst_path: Path, timeout: int = 60) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python urllib"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dst_path.write_bytes(data)


def ensure_cn_font(font_dir: str | os.PathLike = "./fonts") -> Path:
    """
    确保本地有一个可用的中文字体文件，返回字体文件路径。
    """
    font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)

    # 你也可以把自己下载好的字体放在这个目录下，脚本会优先复用
    candidates = [
        font_dir / "NotoSansSC-Regular.otf",
        font_dir / "NotoSansSC-Regular.ttf",
        font_dir / "SourceHanSansSC-Regular.otf",
        font_dir / "simhei.ttf",
        font_dir / "msyh.ttc",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000:
            return p

    # 自动下载 NotoSansSC-Regular.otf
    last_err = None
    for url in NOTO_SANS_SC_URLS:
        try:
            dst = font_dir / "NotoSansSC-Regular.otf"
            download_file(url, dst)
            if dst.exists() and dst.stat().st_size > 100_000:
                return dst
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "自动下载中文字体失败。你可以：\n"
        "1) 检查网络是否能访问 GitHub raw 文件；\n"
        "2) 手动下载任意中文字体（ttf/otf/ttc）放到 ./fonts/ 目录；\n"
        f"最后一次错误：{last_err}"
    )


def setup_matplotlib_cn_font(font_dir: str | os.PathLike = "./fonts"):
    """
    注册字体并设置 matplotlib 全局字体。
    返回 FontProperties，方便在 plt.title/xlabel/ylabel 中显式指定。
    """
    font_path = ensure_cn_font(font_dir)

    # 注册字体到 matplotlib（无需安装到系统字体目录）
    font_manager.fontManager.addfont(str(font_path))
    font_prop = font_manager.FontProperties(fname=str(font_path))

    # 全局设置：让 matplotlib 默认使用该字体
    mpl.rcParams["font.family"] = font_prop.get_name()
    mpl.rcParams["axes.unicode_minus"] = False

    return font_prop, font_path


def _self_test():
    font_prop, font_path = setup_matplotlib_cn_font("./data/fonts")
    print("字体已就绪：", font_path)
    print("matplotlib 使用字体：", mpl.rcParams["font.family"])

    # 简单画一张图测试中文
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 3), dpi=150)
    plt.title("中文测试：特征重要性", fontproperties=font_prop)
    plt.xlabel("横轴：重要性", fontproperties=font_prop)
    plt.ylabel("纵轴：特征", fontproperties=font_prop)
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.tight_layout()
    plt.savefig("cn_font_test.png")
    print("已生成 cn_font_test.png")


if __name__ == "__main__":
    _self_test()