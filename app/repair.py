"""连通组补配区域评估：按指定版本快照评估锚片所在组的补配缺口。

修复师在**任一装配版本**中选一片陶片作为**组锚**，并给出世界坐标
（毫米制）的**目标器形闭合轮廓**，系统评估该片所在连通组的补配区域；
孤片按单片计算：

- 按所选版本快照的固定轮廓与位姿，把组内全部陶片放置到世界坐标；
- 先把每片与目标范围求交（裁入目标范围），再取几何并集——
  碎片超出目标的部分不计，碎片彼此重叠的部分只扣一次；
- 目标减去并集得到缺口；保留缺口的孔洞拓扑：目标范围内若存在独立的
  已覆盖岛（被缺口包围的覆盖区），缺口多边形带内孔；
- 仅共点的缺口不做任何合并（不 buffer / 不 dissolve），按面积降序、
  中心坐标升序逐条返回外边界、内孔边界与面积；
- 目标轮廓自交、锚片在该版不存在、坐标非有限数时整次拒绝并指出原因；
- 无正面积缺口时返回空清单与零缺口面积。

本模块只做**只读**演算：不修改当前装配、关系与快照，也不产生新版本。
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity

from . import geometry

# 正面积缺口判定容差（mm²）：小于该值的退化面（数值零面积薄片/共点面）
# 不视为正面积缺口
POSITIVE_AREA_TOL = 1e-9


class RepairError(Exception):
    """请求级错误（整次拒绝），status 为对应 HTTP 状态码。

    errors 为逐字段错误明细：[{"field", "code", "message"}, ...]。
    """

    def __init__(self, status: int, code: str, message: str,
                 errors: list[dict] | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.errors = errors or []
        self.extra = extra or {}

    def detail(self) -> dict:
        d = {"message": self.message, "code": self.code}
        if self.errors:
            d["errors"] = self.errors
        d.update(self.extra)
        return d


# ---------------------------------------------------------------- 请求校验

def validate_anchor_id(anchor_id: int) -> int:
    """锚片编号必须为正整数（与碎片/间隙接口的编号约定一致）。"""
    if not isinstance(anchor_id, int) or isinstance(anchor_id, bool) or anchor_id < 1:
        raise RepairError(
            400, "invalid_anchor", "锚片编号必须为正整数，整次拒绝",
            errors=[{"field": "anchor_fragment_id", "code": "invalid_anchor",
                     "message": f"无效编号：锚片 id 必须为正整数（当前为 {anchor_id!r}）"}])
    return anchor_id


def validate_target_contour(points) -> list[tuple[float, float]]:
    """校验目标器形轮廓（世界坐标、毫米制）：

    - 必须是 ≥3 个点的闭合轮廓，每点恰为 2 个数值；
    - 所有坐标必须是有限数（NaN / Infinity / -Infinity 整次拒绝，逐点指出）；
    - 轮廓面积非零、无自交（自交/退化整次拒绝并指出原因）。
    """
    if not isinstance(points, (list, tuple)):
        raise RepairError(
            400, "invalid_target_contour", "目标轮廓必须是点坐标数组，整次拒绝",
            errors=[{"field": "target_contour", "code": "invalid_target_contour",
                     "message": "目标轮廓必须是 [[x, y], ...] 形式的数组"}])
    if len(points) < 3:
        raise RepairError(
            400, "invalid_target_contour",
            "目标器形闭合轮廓至少需要 3 个顶点，整次拒绝",
            errors=[{"field": "target_contour", "code": "too_few_points",
                     "message": f"闭合轮廓至少需要 3 个顶点（当前 {len(points)} 个）"}])
    errors: list[dict] = []
    pts: list[tuple[float, float]] = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            errors.append({
                "field": f"target_contour[{i}]", "code": "invalid_point",
                "message": f"第 {i + 1} 个顶点必须是 [x, y] 形式的两个数值"})
            continue
        x, y = p[0], p[1]
        if not isinstance(x, (int, float)) or isinstance(x, bool) \
                or not isinstance(y, (int, float)) or isinstance(y, bool):
            errors.append({
                "field": f"target_contour[{i}]", "code": "non_numeric",
                "message": f"第 {i + 1} 个顶点坐标必须是数值（毫米）"})
            continue
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            errors.append({
                "field": f"target_contour[{i}]", "code": "non_finite_coordinate",
                "message": f"第 {i + 1} 个顶点坐标非有限数"
                           f"（{x!r}, {y!r}）：NaN / Infinity 不允许"})
            continue
        pts.append((float(x), float(y)))
    if errors:
        raise RepairError(
            400, "non_finite_coordinate",
            "目标轮廓存在非有限数或非法坐标，整次拒绝（未执行任何计算）",
            errors=errors)
    # 去闭合重复点 / 连续重复点后再做几何校验
    cleaned = geometry.normalize_contour(pts)
    if len(cleaned) < 3:
        raise RepairError(
            400, "invalid_target_contour",
            "目标轮廓有效顶点不足 3 个（存在重复点），整次拒绝",
            errors=[{"field": "target_contour", "code": "too_few_points",
                     "message": "轮廓有效顶点不足 3 个（存在重复点）"}])
    poly = Polygon(cleaned)
    if not poly.is_valid:
        reason = explain_validity(poly)
        code = "self_intersecting_contour"
        if poly.area <= geometry.CLOSE_TOL:
            message = f"目标轮廓无效（自交或顶点共线导致面积为零）：{reason}"
        else:
            message = f"目标轮廓自交：{reason}"
        raise RepairError(
            400, code, "目标轮廓自交或无效，整次拒绝（未执行任何计算）",
            errors=[{"field": "target_contour", "code": code,
                     "message": message}],
            extra={"reason": reason})
    if poly.area <= POSITIVE_AREA_TOL:
        raise RepairError(
            400, "invalid_target_contour",
            "目标轮廓面积为零，整次拒绝（未执行任何计算）",
            errors=[{"field": "target_contour", "code": "zero_area",
                     "message": "目标器形轮廓面积必须为正（毫米制）"}])
    return cleaned


# ---------------------------------------------------------------- 组定位

def group_of_anchor(snap: dict, anchor_id: int) -> list[dict]:
    """在版本快照中定位锚片所在连通组（孤片按单片计算）。

    连通组按该快照碎片的 group_id 重算（与历史查看/恢复后的分组口径一致，
    不依赖快照里存的 groups 字段）。锚片在该版不存在 → RepairError 404。
    """
    frags = {int(f["id"]): f for f in snap["fragments"]}
    if anchor_id not in frags:
        raise RepairError(
            404, "anchor_not_found",
            f"锚片 #{anchor_id} 在版本 v{snap['version']} 快照中不存在，整次拒绝",
            errors=[{"field": "anchor_fragment_id", "code": "anchor_not_found",
                     "message": f"锚片 #{anchor_id} 未出现在版本 v{snap['version']} "
                                "的碎片清单中（可能在该版本之后才录入或已删除）"}],
            extra={"version": snap["version"], "anchor_fragment_id": anchor_id})
    anchor = frags[anchor_id]
    gid = anchor.get("group_id")
    if gid is None:
        return [anchor]
    return sorted((f for f in snap["fragments"] if f.get("group_id") == gid),
                  key=lambda f: int(f["id"]))


# ---------------------------------------------------------------- 补配演算

def _signed_area(ring) -> float:
    s = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _splice_slit(outer, hole, pinch):
    """把与外环仅在 pinch 点共点的内孔（狭缝孔）折进外环，得到自触外环。

    覆盖物仅以一点接触目标边界时，Shapely 把「该点两侧实为连通的单一
    区域」编码为一个外环 + 与外环共点的内孔（如从边界垂下的悬空墙两侧
    在墙底连通，是一个 U 形单面而非两个缺口）。把该狭缝孔按拓扑展开进
    外环后：区域面积不变（狭缝零宽）、区域仍为一个缺口，且该覆盖物不是
    被覆盖区包围的「独立岛」，故不再作为内孔报告。
    """
    def rot(ring, pt):
        i = ring.index(pt)
        return ring[i:] + ring[:i]

    o = rot(outer, pinch)
    h = rot(hole, pinch)
    # 外环与狭缝孔同方向时，孔环需反向后再拼接（保证面积 = |外环| - |孔|）
    if _signed_area(o) * _signed_area(h) > 0:
        h = list(reversed(h))
    return o + h


def _normalize_polygon(poly: Polygon) -> Polygon:
    """把「狭缝孔」（与外环共点的内孔）展开进外环；被覆盖区真正包围的
    独立岛保留为内孔（缺口孔洞拓扑）。
    """
    outer = [tuple(p) for p in poly.exterior.coords[:-1]]
    keep_holes = []
    ring = outer
    for r in poly.interiors:
        hp = [tuple(p) for p in r.coords[:-1]]
        common = [q for q in hp if q in set(ring)]
        if len(common) == 1:
            ring = _splice_slit(ring, hp, common[0])   # 狭缝：折进外环
        else:
            keep_holes.append(hp)                       # 真岛（或 ≥2 共点）
    if ring == outer:
        return poly
    return Polygon(ring, keep_holes)


def _polygons(geom):
    """从任意平面几何中提取全部正面积 Polygon（递归 GeometryCollection）。"""
    out = []
    if geom.geom_type == "Polygon":
        if not geom.is_valid:
            geom = geom.buffer(0)
            return _polygons(geom)
        if geom.area > POSITIVE_AREA_TOL:
            out.append(_normalize_polygon(geom))
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            out.extend(_polygons(g))
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            out.extend(_polygons(g))
    # Point / MultiPoint / LineString / MultiLineString 无面积，忽略
    return out


def assess_repair_region(snap: dict, anchor_id: int,
                         target_pts) -> dict:
    """在版本快照 snap 上执行补配区域评估（纯只读演算，不写库）。

    返回未舍入的演算结果：
        target_area / covered_area / gap_area、缺口多边形列表（含外边界、
        内孔边界、面积与质心）、组内碎片清单、锚片信息。
    """
    members = group_of_anchor(snap, anchor_id)
    target = Polygon(target_pts)
    target_area = target.area

    # 按该版轮廓与位姿放置组内全部陶片；先裁入目标范围，再取几何并集。
    # 超出目标的部分在求交时被裁掉；组内重叠由并集天然只计一次。
    clipped: list[Polygon] = []
    placements: list[dict] = []
    for f in members:
        wc = geometry.contour_world(f["contour"],
                                    (f["x"], f["y"], f["theta"]))
        poly = Polygon(wc)
        if not poly.is_valid:
            poly = poly.buffer(0)
        part = poly.intersection(target)
        part_area = part.area
        if part_area > POSITIVE_AREA_TOL:
            clipped.extend(_polygons(part))
        placements.append({
            "fragment_id": int(f["id"]),
            "name": f["name"],
            "anchor": int(f["id"]) == anchor_id,
            "clipped_area_mm2": part_area,
        })
    covered = unary_union(clipped) if clipped else Polygon()
    covered_area = min(covered.area, target_area)

    # 目标减去并集得到缺口。difference 不合并仅共点的相邻面：
    # 结果中每个正面积 Polygon 即一条独立缺口，共点连接保持分离。
    gap_geom = target.difference(covered)
    gap_polys = _polygons(gap_geom)

    gaps: list[dict] = []
    for gp in gap_polys:
        c = gp.centroid
        gaps.append({
            "area_mm2": gp.area,
            "centroid": (c.x, c.y),
            "exterior": list(gp.exterior.coords)[:-1],   # 去掉闭合重复末点
            "holes": [list(r.coords)[:-1] for r in gp.interiors],
        })
    # 面积降序；面积相同按中心坐标 (x, y) 升序
    gaps.sort(key=lambda g: (-g["area_mm2"], g["centroid"][0], g["centroid"][1]))
    gap_area = sum(g["area_mm2"] for g in gaps)

    return {
        "version": int(snap["version"]),
        "anchor_id": anchor_id,
        "group_fragment_ids": [int(f["id"]) for f in members],
        "independent": len(members) == 1,
        "placements": placements,
        "target_contour": target_pts,
        "target_area": target_area,
        "covered_area": covered_area,
        "gap_area": gap_area,
        "gaps": gaps,
    }


# ---------------------------------------------------------------- 响应整理

def _r(v: float, ndigits: int = 6) -> float:
    v = round(v, ndigits)
    return 0.0 if v == 0 else v   # 归一 -0.0


def _ring(points) -> list[list[float]]:
    return [[_r(x, 6), _r(y, 6)] for x, y in points]


def format_result(res: dict) -> dict:
    """舍入并规范化评估响应（面积 mm²、轮廓点坐标 mm）。"""
    gaps = []
    for i, g in enumerate(res["gaps"]):
        cx, cy = g["centroid"]
        gaps.append({
            "index": i + 1,
            "area_mm2": _r(g["area_mm2"]),
            "centroid": [_r(cx), _r(cy)],
            "exterior": _ring(g["exterior"]),
            "holes": [_ring(h) for h in g["holes"]],
            "hole_count": len(g["holes"]),
        })
    return {
        "ok": True,
        "version": res["version"],
        "anchor_fragment_id": res["anchor_id"],
        "group_fragment_ids": res["group_fragment_ids"],
        "independent": res["independent"],
        "target_contour": _ring(res["target_contour"]),
        "target_area_mm2": _r(res["target_area"]),
        "covered_area_mm2": _r(res["covered_area"]),
        "gap_total_area_mm2": _r(res["gap_area"]),
        "gap_count": len(gaps),
        "gaps": gaps,
        "fragments": [
            {"fragment_id": p["fragment_id"], "name": p["name"],
             "anchor": p["anchor"],
             "clipped_area_mm2": _r(p["clipped_area_mm2"])}
            for p in res["placements"]
        ],
    }
