"""Draw the illustrative residual-stream figure in desktop and mobile layouts."""

import subprocess
from functools import cache
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

BLUE = "#4779a8"
ORANGE = "#b96243"
INK = "#252525"
MUTED = "#6b6b6b"
CONTEXT_ACTIVATIONS = [
    [0.15, 0.70, 0.40, 0.85, 0.25, 0.55],
    [0.60, 0.20, 0.75, 0.30, 0.50, 0.85],
    [0.80, 0.45, 0.20, 0.60, 0.90, 0.30],
    [0.40, 0.85, 0.55, 0.15, 0.65, 0.75],
]
ANSWER_ACTIVATIONS = [
    [0.80, 0.35, 0.60, 0.20, 0.75, 0.45],
    [0.60, 0.45, 0.80, 0.30, 0.55, 0.65],
]


@cache
def render_latex(*, text):
    document = (
        r"\documentclass{standalone}"
        "\n"
        r"\usepackage{amsmath,amssymb}"
        "\n"
        r"\begin{document}\sffamily"
        "\n"
        r'\special{dvisvgm:raw <g id="label" data-baseline="{?y}">}'
        + text
        + r"\special{dvisvgm:raw </g>}\end{document}"
    )
    with TemporaryDirectory(prefix="activation-label-") as directory:
        path = Path(directory)
        (path / "label.tex").write_text(document, encoding="utf-8")
        for command in [
            ["latex", "-interaction=nonstopmode", "-halt-on-error", "label.tex"],
            ["dvisvgm", "--no-fonts=1", "--output=label.svg", "label.dvi"],
        ]:
            result = subprocess.run(
                command, cwd=path, capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        root = ElementTree.parse(path / "label.svg").getroot()

    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    label = root.find('.//{http://www.w3.org/2000/svg}g[@id="label"]')
    left, _, width, _ = map(float, root.attrib["viewBox"].split())
    baseline = float(label.attrib["data-baseline"])
    paths = "".join(ElementTree.tostring(child, encoding="unicode") for child in label)
    return paths, left, baseline, width


class Diagram:
    def __init__(self, *, width, height):
        self.parts = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">'
            ),
            '<title id="title">From context residuals to a distribution of answer means</title>',
            (
                '<desc id="description">Context and answer tokens are at the bottom of a '
                "language model. Continuous upward paths carry the residual streams; left branches "
                "show layer updates joining through circled-plus addition nodes. Context residuals after "
                "layer ell are blue and answer residuals at the same layer are orange. "
                "The context trace feeds a learned metamodel. Averaging the answer token "
                "vectors gives one target vector for this rollout. Other possible answers "
                "are stacked as y2, y3, and an ellipsis below y1, grouped by a large brace. Their answer means are forecast before "
                "generation. Tokens, token boundaries, and activation values are schematic.</desc>"
            ),
            "<defs>",
        ]
        for name, color in [("blue", BLUE), ("orange", ORANGE), ("gray", "#a9a9a9")]:
            self.parts.append(
                f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="8" refY="5" '
                f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        self.parts.extend(
            [
                "</defs>",
                '<g font-family="Arial, Helvetica, sans-serif" fill="#252525">',
                f'<rect width="{width}" height="{height}" fill="white"/>',
            ]
        )

    def text(self, *, x, y, text, size=16, color=INK, anchor="start", weight=400):
        self.parts.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>'
        )

    def latex(self, *, x, y, text, size=16, color=INK, anchor="start"):
        paths, left, baseline, width = render_latex(text=text)
        # dvisvgm uses 1/72-inch units; the source font is 10 TeX points (1/72.27 inch).
        scale = size / 10 * 72.27 / 72
        offset = {"start": 0, "middle": 0.5, "end": 1}[anchor] * width * scale
        self.parts.append(
            f'<g fill="{color}" aria-label="{escape(text)}" '
            f'transform="translate({x - offset:g} {y:g}) scale({scale:g}) '
            f'translate({-left:g} {-baseline:g})">{paths}</g>'
        )

    def box(self, *, x, y, width, height, fill="white", stroke="none", radius=5):
        self.parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>'
        )

    def path(self, *, points, color, arrow=None, dashed=False, width=1.5):
        attributes = f' marker-end="url(#arrow-{arrow})"' if arrow else ""
        if dashed:
            attributes += ' stroke-dasharray="5 5"'
        self.parts.append(
            f'<path d="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{attributes}/>'
        )

    def vector(self, *, x, y, values, color, width=22):
        self.box(x=x - width / 2, y=y, width=width, height=48, fill="white", radius=2)
        for index, value in enumerate(values):
            self.parts.append(
                f'<rect x="{x - width / 2:g}" y="{y + index * 8:g}" '
                f'width="{width}" height="7" fill="{color}" fill-opacity="{value:.3f}"/>'
            )
        self.box(
            x=x - width / 2,
            y=y,
            width=width,
            height=48,
            fill="none",
            stroke=color,
            radius=2,
        )

    def residual_layer(self, *, position, block_y, mobile):
        branch_x = position - (32 if mobile else 48)
        block_width = 22 if mobile else 36
        radius = 6.5 if mobile else 8
        addition_y = block_y - 12
        source_y = block_y + 44
        self.path(
            points=f"M {position} {source_y} H {branch_x} V {block_y + 34}",
            color="#969b9f",
            arrow="gray",
            width=1.3,
        )
        self.box(
            x=branch_x - block_width / 2,
            y=block_y,
            width=block_width,
            height=32,
            fill="#f3f4f5",
            stroke="#a7acb0",
            radius=3,
        )
        self.path(
            points=f"M {branch_x} {block_y - 1} V {addition_y} H {position - radius - 2}",
            color="#969b9f",
            arrow="gray",
            width=1.3,
        )
        self.parts.append(
            f'<circle cx="{position}" cy="{source_y}" r="2" fill="#969b9f"/>'
        )
        self.parts.append(
            f'<circle cx="{position}" cy="{addition_y}" r="{radius}" '
            'fill="white" stroke="#727a80" stroke-width="1.4"/>'
        )
        self.path(
            points=f"M {position - 3.5} {addition_y} H {position + 3.5} "
            f"M {position} {addition_y - 3.5} V {addition_y + 3.5}",
            color="#727a80",
            width=1.3,
        )

    def token(self, *, position, y, text, color, mobile):
        token_width = 42 if mobile else 62
        self.box(
            x=position - token_width / 2,
            y=y,
            width=token_width,
            height=34,
            fill="#f5f8fb" if color == BLUE else "#fcf5f1",
            stroke="#c4d2df" if color == BLUE else "#dfbcae",
        )
        self.text(
            x=position, y=y + 23, text=text, size=14 if mobile else 17, anchor="middle"
        )

    def language_model(self, *, mobile):
        context_positions = [90, 146, 202, 258] if mobile else [210, 300, 390, 480]
        answer_positions = [348, 408] if mobile else [670, 760]
        vector_width = 12 if mobile else 16
        title_x = 20 if mobile else 145
        self.text(x=title_x, y=31, text="Residual streams", size=24, weight=600)

        for positions, color, fill in [
            (context_positions, BLUE, "#edf4fa"),
            (answer_positions, ORANGE, "#fcf1eb"),
        ]:
            left = positions[0] - (32 + 11 + 8 if mobile else 48 + 18 + 8)
            right = positions[-1] + 12 + vector_width / 2 + 8
            self.box(
                x=left,
                y=233,
                width=right - left,
                height=66,
                fill=fill,
                stroke=color,
                radius=4,
            )

        for label, block_y in [("L", 144), (r"\ell", 352), ("1", 532)]:
            self.latex(
                x=44 if mobile else 125,
                y=block_y + 22,
                text=f"layer ${label}$",
                size=11 if mobile else 18,
                anchor="end",
                color=MUTED,
            )
        for label, baseline in [("residual", 259), ("after $x$", 280)]:
            self.latex(
                x=36 if mobile else 125,
                y=baseline,
                text=label,
                size=11 if mobile else 17,
                anchor="end",
                color=MUTED,
            )

        streams = list(
            zip(context_positions, ["Name", "a", "fruit", "."], CONTEXT_ACTIVATIONS)
        )
        streams.extend(zip(answer_positions, ["Apple", "."], ANSWER_ACTIVATIONS))
        for index, (position, token, values) in enumerate(streams):
            color = BLUE if index < len(context_positions) else ORANGE
            self.path(
                points=f"M {position} 592 V 74",
                color="#727a80",
                arrow="gray",
                width=1.8,
            )
            for block_y in [532, 352, 144]:
                self.residual_layer(position=position, block_y=block_y, mobile=mobile)
            for baseline in [218, 450]:
                self.text(
                    x=position - (32 if mobile else 48),
                    y=baseline,
                    text="⋮",
                    size=23,
                    color="#a7acb0",
                    anchor="middle",
                )
            for vector_y, vector_color, vector_values in [
                (428, "#949a9e", list(reversed(values))),
                (242, color, values),
            ]:
                self.path(
                    points=f"M {position} {vector_y + 24} H {position + 12 - vector_width / 2}",
                    color=vector_color,
                    width=1.3,
                )
                self.vector(
                    x=position + 12,
                    y=vector_y,
                    values=vector_values,
                    color=vector_color,
                    width=vector_width,
                )
            if index < len(context_positions):
                self.token(
                    position=position, y=600, text=token, color=color, mobile=mobile
                )

        self.path(
            points=f"M {context_positions[0] - 20} 648 V 656 H {context_positions[-1] + 20} V 648",
            color=BLUE,
        )
        self.latex(
            x=(context_positions[0] + context_positions[-1]) / 2,
            y=682,
            text="Context $x$",
            size=17 if mobile else 20,
            color=BLUE,
            anchor="middle",
        )

        brace_tip = 298 if mobile else 590
        brace_stem = 290 if mobile else 578
        brace_middle = 282 if mobile else 566
        self.path(
            points=(
                f"M {brace_tip} 594 "
                f"C {brace_stem} 594 {brace_stem} 602 {brace_stem} 618 "
                f"V 658 C {brace_stem} 674 {brace_stem} 682 {brace_middle} 682 "
                f"C {brace_stem} 682 {brace_stem} 690 {brace_stem} 706 "
                f"V 746 C {brace_stem} 762 {brace_stem} 770 {brace_tip} 770"
            ),
            color=ORANGE,
            width=2,
        )
        for index, word in enumerate(["Apple", "Pear", "Plum"]):
            row_y = 600 + index * 48
            self.latex(
                x=answer_positions[0] - (30 if mobile else 46),
                y=row_y + 24,
                text=f"$y_{index + 1}$",
                size=18 if mobile else 21,
                color=ORANGE,
                anchor="end",
            )
            for position, token in zip(answer_positions, [word, "."]):
                self.token(
                    position=position, y=row_y, text=token, color=ORANGE, mobile=mobile
                )
        answer_center = sum(answer_positions) / len(answer_positions)
        self.text(
            x=answer_center, y=761, text="⋮", size=26, color=ORANGE, anchor="middle"
        )
        self.latex(
            x=answer_center,
            y=794,
            text=r"$y \sim p(y \mid x)$",
            size=13 if mobile else 16,
            color=MUTED,
            anchor="middle",
        )

    def answer_mean(self, *, x, y):
        values = ANSWER_ACTIVATIONS
        mean = [sum(dimension) / len(values) for dimension in zip(*values)]
        self.vector(x=x, y=y, values=mean, color=ORANGE)

    def distribution(self, *, x, y, width):
        center = x + width / 2
        self.parts.append(
            f'<ellipse cx="{center}" cy="{y + 65}" rx="{width * 0.44}" ry="61" '
            f'fill="#edf4fa" stroke="{BLUE}" stroke-width="1.5"/>'
        )
        for radius, opacity in [(0.32, 0.5), (0.19, 0.3)]:
            self.parts.append(
                f'<ellipse cx="{center}" cy="{y + 65}" rx="{width * radius}" '
                f'ry="{radius * 120}" fill="none" stroke="{BLUE}" stroke-opacity="{opacity}"/>'
            )
        for index, (offset_x, offset_y) in enumerate(
            [(-0.22, 10), (0.08, -25), (0.25, 25)]
        ):
            point_x, point_y = center + width * offset_x, y + 65 + offset_y
            self.parts.append(
                f'<circle cx="{point_x}" cy="{point_y}" r="5.5" fill="{ORANGE}"/>'
            )
            self.text(
                x=point_x + 11,
                y=point_y + 5,
                text=str(index + 1),
                size=14,
                color=ORANGE,
            )
        self.path(
            points=f"M {center - 7} {y + 65} H {center + 7} M {center} {y + 58} V {y + 72}",
            color=BLUE,
            width=2.5,
        )
        self.text(
            x=center,
            y=y + 154,
            text="Distribution of answer means",
            size=18,
            anchor="middle",
            weight=600,
        )
        self.text(
            x=center,
            y=y + 179,
            text="One orange point per sampled answer",
            size=14,
            color=MUTED,
            anchor="middle",
        )

    def save(self, *, path):
        self.parts.extend(["</g>", "</svg>"])
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def build_diagram(*, mobile):
    width, height = (440, 1304) if mobile else (1260, 820)
    diagram = Diagram(width=width, height=height)
    diagram.language_model(mobile=mobile)

    if mobile:
        diagram.parts.append('<g transform="translate(0 64)">')
        diagram.path(points="M 20 758 H 420", color="#e4e4e4")
        diagram.latex(
            x=20, y=788, text=r"\textbf{Forecast at the same layer $\ell$}", size=20
        )
        diagram.latex(x=32, y=839, text=r"$C_\ell(x)$", size=23, color=BLUE)
        diagram.path(points="M 98 831 H 161", color=BLUE, arrow="blue", width=2)
        diagram.box(x=173, y=805, width=247, height=54, fill="#f1f6fa", stroke=BLUE)
        diagram.text(
            x=296, y=839, text="Learned metamodel", size=18, anchor="middle", weight=600
        )
        diagram.text(
            x=20,
            y=887,
            text="Mean over the sampled answer’s tokens",
            size=16,
            color=MUTED,
        )
        for position, values in zip([44, 81], ANSWER_ACTIVATIONS):
            diagram.vector(x=position, y=906, values=values, color=ORANGE, width=16)
        diagram.path(points="M 103 930 H 165", color=ORANGE, arrow="orange")
        diagram.answer_mean(x=190, y=906)
        diagram.latex(x=220, y=937, text=r"$A_\ell(x,s_1)$", size=22, color=ORANGE)
        diagram.path(
            points="M 296 860 V 871 H 430 V 993 H 330 V 1023",
            color=BLUE,
            arrow="blue",
            width=2,
        )
        diagram.distribution(x=30, y=1010, width=380)
        diagram.path(
            points="M 190 957 V 987 H 137 V 1071",
            color=ORANGE,
            arrow="orange",
            dashed=True,
        )
        diagram.text(
            x=220,
            y=1220,
            text="Forecast before any answer is generated",
            size=15,
            color=MUTED,
            anchor="middle",
        )
        diagram.parts.append("</g>")
    else:
        diagram.path(
            points="M 508 266 H 545 V 62 H 1040 V 104",
            color=BLUE,
            arrow="blue",
            width=2,
        )
        diagram.latex(
            x=890, y=47, text=r"Context trace $C_\ell(x)$", size=19, color=BLUE
        )
        diagram.box(x=890, y=108, width=300, height=66, fill="#f1f6fa", stroke=BLUE)
        diagram.text(
            x=1040,
            y=147,
            text="Learned metamodel",
            size=20,
            anchor="middle",
            weight=600,
        )
        diagram.path(points="M 1040 176 V 506", color=BLUE, arrow="blue", width=2)
        diagram.text(x=1057, y=465, text="forecast", size=17, color=BLUE)
        diagram.path(
            points="M 788 266 H 830 V 335 H 891", color=ORANGE, arrow="orange", width=2
        )
        diagram.text(
            x=908, y=272, text="Mean over", size=17, color=MUTED, anchor="middle"
        )
        diagram.text(
            x=908, y=295, text="answer tokens", size=17, color=MUTED, anchor="middle"
        )
        diagram.answer_mean(x=908, y=311)
        diagram.latex(
            x=908,
            y=388,
            text=r"$A_\ell(x,s_1)$",
            size=22,
            color=ORANGE,
            anchor="middle",
        )
        diagram.distribution(x=860, y=510, width=360)
        diagram.path(
            points="M 908 399 V 475 H 961 V 571",
            color=ORANGE,
            arrow="orange",
            dashed=True,
        )
        diagram.text(
            x=1040,
            y=735,
            text="Forecast before generation",
            size=16,
            color=MUTED,
            anchor="middle",
        )

    return diagram


if __name__ == "__main__":
    asset_directory = Path(__file__).resolve().parent / "asset"
    for mobile, filename in [
        (False, "activation-level.svg"),
        (True, "activation-level-mobile.svg"),
    ]:
        build_diagram(mobile=mobile).save(path=asset_directory / filename)
