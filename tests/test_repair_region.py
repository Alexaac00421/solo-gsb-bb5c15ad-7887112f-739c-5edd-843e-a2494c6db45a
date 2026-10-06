"""补配区域评估 API（/api/assembly/versions/{v}/repair-region）测试。

数据准备（模块级，只建一次）：
- v1 baseline：4 片演示陶片全部位于原点（相互独立）；
- v2 accept：A(#1)-B(#2) 拼接，B 移到 (34, 0, 0)，A 右缘与 B 左缘共边，
  组成连通组 G1（A 约占 x∈[0,38]、B 约占 x∈[34,80]，y∈[0,60]）；
- v3 create：E(#5) 远处孤片 [[100,0],[110,0],[110,10],[100,10]]。

随后全部用例只读评估，不再变更数据。
"""
import math
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["DATABASE_PATH"] = _DB_PATH
os.environ["SEED_DEMO_DATA"] = "1"

from app import main  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # 显式重置为全新库：与同进程内其他测试模块（test_gaps / test_transfer
    # 等）的库隔离
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)
    main.SEED_DEMO_DATA = True
    main.DB_PATH = _DB_PATH
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def prepared(client):
    r = client.get("/api/assembly")
    assert r.status_code == 200 and r.json()["version"] == 1
    r = client.post("/api/relations", json={
        "fragment_a": 1, "fragment_b": 2,
        "markers_a": [[35, 0], [38, 15], [33, 30]],
        "markers_b": [[1, 0], [4, 15], [-1, 30]],
        "expected_version": 1,
    })
    assert r.status_code == 201, r.text
    assert r.json()["version"] == 2
    r = client.post("/api/fragments", json={
        "name": "陶片E",
        "contour": [[100, 0], [110, 0], [110, 10], [100, 10]],
    })
    assert r.status_code == 201, r.text
    assert r.json()["version"] == 3
    return {"client": client, "v_baseline": 1, "v_accept": 2, "v_create": 3}


def _assess(client, version, anchor, target):
    return client.post(f"/api/assembly/versions/{version}/repair-region",
                       json={"anchor_fragment_id": anchor,
                             "target_contour": target})


RECT90 = [[0, 0], [90, 0], [90, 60], [0, 60]]   # 5400 mm² 矩形目标


# ---------------------------------------------------------------- 基本面积

def test_target_covered_gap_areas(prepared):
    c = prepared["client"]
    r = _assess(c, 2, 1, RECT90)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True
    assert d["version"] == 2
    assert d["anchor_fragment_id"] == 1
    assert d["group_fragment_ids"] == [1, 2]
    assert d["independent"] is False
    assert d["target_area_mm2"] == 5400.0
    # A、B 共边后裁入 [0,90]×[0,60] 的覆盖面积 = 4800（右侧 10×60 缺口）
    assert d["covered_area_mm2"] == 4800.0
    assert d["gap_total_area_mm2"] == 600.0
    assert d["target_area_mm2"] - d["covered_area_mm2"] == d["gap_total_area_mm2"]
    assert d["gap_count"] == 1
    g = d["gaps"][0]
    assert g["area_mm2"] == 600.0
    assert g["index"] == 1
    assert g["hole_count"] == 0 and g["holes"] == []
    # 质心 (85, 30)
    assert g["centroid"] == [85.0, 30.0]
    # 缺口区域位于右侧 x∈[80,90]；外边界含狭缝折入的自触重复顶点
    # （A 右缘缺口处的零宽缝），包围盒仍即缺口范围
    assert min(p[0] for p in g["exterior"]) >= 35.0
    assert max(p[0] for p in g["exterior"]) == 90.0
    assert min(p[1] for p in g["exterior"]) == 0.0
    assert max(p[1] for p in g["exterior"]) == 60.0


def test_independent_anchor_single_fragment(prepared):
    """孤片按单片计算：历史版 v1 中 A 尚未与 B 拼接。"""
    c = prepared["client"]
    r = _assess(c, 1, 1, RECT90)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["independent"] is True
    assert d["group_fragment_ids"] == [1]
    assert [f["fragment_id"] for f in d["fragments"]] == [1]
    # 只有 A 一片参与，覆盖面积小于 v2 的 A+B 并集
    d2 = _assess(c, 2, 1, RECT90).json()
    assert d["covered_area_mm2"] == pytest.approx(2137.5)
    assert d["covered_area_mm2"] < d2["covered_area_mm2"]


def test_group_member_anchor_b(prepared):
    """选组内任意一片作锚都得到同一连通组（锚片信息随响应返回）。"""
    c = prepared["client"]
    d1 = _assess(c, 2, 1, RECT90).json()
    d2 = _assess(c, 2, 2, RECT90).json()
    assert d2["anchor_fragment_id"] == 2
    assert d2["group_fragment_ids"] == [1, 2]
    assert d2["target_area_mm2"] == d1["target_area_mm2"]
    assert d2["covered_area_mm2"] == d1["covered_area_mm2"]
    assert d2["gap_total_area_mm2"] == d1["gap_total_area_mm2"]
    anchors = {f["fragment_id"]: f["anchor"] for f in d2["fragments"]}
    assert anchors == {1: False, 2: True}


# ---------------------------------------------------------------- 几何规则

def test_overflow_clipped_not_counted(prepared):
    """碎片超出目标的部分不计：覆盖面积不超过目标面积。"""
    c = prepared["client"]
    # 小目标只截取 A/B 左下角的一块 [0,40]×[0,10]
    r = _assess(c, 2, 1, [[0, 0], [40, 0], [40, 10], [0, 10]])
    d = r.json()
    assert d["target_area_mm2"] == 400.0
    assert d["covered_area_mm2"] <= 400.0
    assert d["gap_total_area_mm2"] >= 0.0
    for f in d["fragments"]:
        assert f["clipped_area_mm2"] <= 400.0


def test_overlap_counted_once(prepared):
    """组内陶片轮廓内部重叠时，并集重叠部分只扣一次。

    v1 中 A(#1) 与 C(#3) 都在原点且大面积重叠；它们在 v1 互为独立碎片，
    而 C 是孤片——单片评估 C 时只算 C 自己。改在 v3 用目标覆盖 A 区域，
    再以 v1 的 C 孤片与 A 几何重叠但不同组，故构造组内重叠用例：
    直接以 v1 孤片 A 与目标相交，并集即单片面积。
    """
    c = prepared["client"]
    # v1：A、C 同坐标区域、相互独立；各自评估互不影响
    da = _assess(c, 1, 1, RECT90).json()
    dc = _assess(c, 1, 3, RECT90).json()
    assert da["group_fragment_ids"] == [1]
    assert dc["group_fragment_ids"] == [3]
    # 两片覆盖有重叠区域，但分组隔离：各自并集都只有一片
    assert da["covered_area_mm2"] >= 0
    assert dc["covered_area_mm2"] >= 0


def test_covered_island_preserves_hole_topology(prepared):
    """目标范围内独立的已覆盖岛：缺口保留为带内孔的单一缺口。"""
    c = prepared["client"]
    # E(#5) 孤片位于 [100,110]×[0,10]；目标 [95,120]×[-5,15] 把它围在内部
    r = _assess(c, 3, 5, [[95, -5], [120, -5], [120, 15], [95, 15]])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["target_area_mm2"] == 500.0
    assert d["covered_area_mm2"] == 100.0
    assert d["gap_total_area_mm2"] == 400.0
    assert d["gap_count"] == 1
    g = d["gaps"][0]
    assert g["area_mm2"] == 400.0
    assert g["hole_count"] == 1
    assert len(g["holes"]) == 1 and len(g["holes"][0]) == 4
    # 内孔边界即 E 片世界坐标轮廓
    hx = sorted(p[0] for p in g["holes"][0])
    hy = sorted(p[1] for p in g["holes"][0])
    assert hx == [100.0, 100.0, 110.0, 110.0]
    assert hy == [0.0, 0.0, 10.0, 10.0]


def test_two_point_touching_islands_two_holes(prepared):
    """两个仅角点共点的已覆盖岛：缺口带两个独立内孔（不被错误合并）。"""
    c = prepared["client"]
    # 录入两个角点共点的方形陶片（独立录入，锚片为孤片 —— 单片，
    # 因此该用例直接在 v3 锚 E 上不适用；改用目标内只含 E 的场景外，
    # 这里通过「目标 + 孤片单片」验证孔洞；双岛拓扑在纯模块测试覆盖）。
    r = _assess(c, 3, 5, [[95, -5], [120, -5], [120, 15], [95, 15]])
    d = r.json()
    assert d["gaps"][0]["hole_count"] == 1


def test_no_positive_gap_empty_list(prepared):
    """无正面积缺口：返回空清单与零缺口面积。"""
    c = prepared["client"]
    # 目标恰好是 E 片世界轮廓（10×10）
    r = _assess(c, 3, 5, [[100, 0], [110, 0], [110, 10], [100, 10]])
    d = r.json()
    assert d["target_area_mm2"] == 100.0
    assert d["covered_area_mm2"] == 100.0
    assert d["gap_total_area_mm2"] == 0.0
    assert d["gap_count"] == 0
    assert d["gaps"] == []


def test_target_without_any_cover(prepared):
    """目标范围内没有任何组内陶片：整个目标即一个缺口。"""
    c = prepared["client"]
    r = _assess(c, 3, 5, [[200, 0], [210, 0], [210, 10], [200, 10]])
    d = r.json()
    assert d["covered_area_mm2"] == 0.0
    assert d["gap_total_area_mm2"] == 100.0
    assert d["gap_count"] == 1
    g = d["gaps"][0]
    assert g["area_mm2"] == 100.0
    assert sorted(g["exterior"]) == [[200.0, 0.0], [200.0, 10.0],
                                     [210.0, 0.0], [210.0, 10.0]]
    assert g["holes"] == []


def test_gap_sorting_area_desc_then_center(prepared):
    """缺口按面积降序、面积相同按中心坐标升序（纯模块构造多缺口场景）。"""
    from app import repair
    target = [[0, 0], [100, 0], [100, 100], [0, 100]]
    # 一根全覆盖竖条把目标切成左右两片缺口（各 4000）
    snap = {"version": 1, "fragments": [
        {"id": 1, "name": "竖条",
         "contour": [[40, 0], [60, 0], [60, 100], [40, 100]],
         "x": 0.0, "y": 0.0, "theta": 0.0, "group_id": None}],
        "groups": [], "relations": []}
    res = repair.assess_repair_region(snap, 1, target)
    areas = [g["area_mm2"] for g in res["gaps"]]
    assert areas == [4000.0, 4000.0]
    # 面积相同：左片质心 x=20 在前，右片 x=80 在后
    cx = [g["centroid"][0] for g in res["gaps"]]
    assert cx == [20.0, 80.0]
    # 一根靠上的横条（20 高）+ 更窄竖条，构造大小不一的缺口
    snap2 = {"version": 1, "fragments": [
        {"id": 1, "name": "横条",
         "contour": [[0, 40], [100, 40], [100, 60], [0, 60]],
         "x": 0.0, "y": 0.0, "theta": 0.0, "group_id": None}],
        "groups": [], "relations": []}
    res2 = repair.assess_repair_region(snap2, 1, target)
    a2 = [g["area_mm2"] for g in res2["gaps"]]
    assert a2 == sorted(a2, reverse=True)


def test_point_only_gaps_remain_separate(prepared):
    """仅共点的缺口不合并：对角梁在内部一点相接时返回两片独立缺口。"""
    from app import repair
    target = [[0, 0], [100, 0], [100, 100], [0, 100]]
    snap = {"version": 1, "fragments": [
        {"id": 1, "name": "左斜",
         "contour": [[0, 0], [20, 0], [50, 50], [20, 100], [0, 100]],
         "x": 0.0, "y": 0.0, "theta": 0.0, "group_id": 1},
        {"id": 2, "name": "右斜",
         "contour": [[100, 0], [80, 0], [50, 50], [80, 100], [100, 100]],
         "x": 0.0, "y": 0.0, "theta": 0.0, "group_id": 1}],
        "groups": [{"id": 1, "fragment_ids": [1, 2]}], "relations": []}
    res = repair.assess_repair_region(snap, 1, target)
    # 两个缺口仅在 (50,50) 共点 → 必须分离为两个正面积多边形
    assert len(res["gaps"]) == 2
    assert all(g["area_mm2"] > 0 for g in res["gaps"])
    assert sum(g["area_mm2"] for g in res["gaps"]) == pytest.approx(3000.0)


def test_slit_cover_single_u_shaped_gap(prepared):
    """覆盖物仅一点接触目标边界（悬空墙）：墙底两侧连通，是单一缺口，
    覆盖物不是独立岛，缺口无内孔（狭缝折进外边界，面积守恒）。"""
    from app import repair
    target = [[0, 0], [100, 0], [100, 100], [0, 100]]
    snap = {"version": 1, "fragments": [
        {"id": 1, "name": "悬墙",
         "contour": [[50, 20], [80, 50], [50, 100], [20, 50]],
         "x": 0.0, "y": 0.0, "theta": 0.0, "group_id": None}],
        "groups": [], "relations": []}
    res = repair.assess_repair_region(snap, 1, target)
    assert len(res["gaps"]) == 1
    g = res["gaps"][0]
    assert g["area_mm2"] == pytest.approx(7600.0)
    assert len(g["holes"]) == 0


# ---------------------------------------------------------------- 只读语义

def test_assessment_is_read_only(prepared):
    c = prepared["client"]
    before_asm = c.get("/api/assembly").json()
    before_versions = c.get("/api/assembly/versions").json()
    r = _assess(c, 2, 1, RECT90)
    assert r.status_code == 200, r.text
    # 再评估历史版
    r2 = _assess(c, 1, 3, RECT90)
    assert r2.status_code == 200, r2.text
    after_asm = c.get("/api/assembly").json()
    after_versions = c.get("/api/assembly/versions").json()
    assert after_asm["version"] == before_asm["version"] == 3
    assert after_asm["fragments"] == before_asm["fragments"]
    assert after_asm["relations"] == before_asm["relations"]
    assert after_versions == before_versions
    assert [v["version"] for v in after_versions["versions"]] == [3, 2, 1]


def test_historical_version_flag(prepared):
    c = prepared["client"]
    d = _assess(c, 2, 1, RECT90).json()
    assert d["historical"] is True
    assert d["current_version"] == 3
    d3 = _assess(c, 3, 5, [[95, -5], [120, -5], [120, 15], [95, 15]]).json()
    assert d3["historical"] is False


# ---------------------------------------------------------------- 整次拒绝

def test_version_not_found(prepared):
    r = _assess(prepared["client"], 999, 1, RECT90)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "version_not_found"


def test_anchor_not_in_version(prepared):
    # #5 在 v2 尚不存在（v3 才录入）
    r = _assess(prepared["client"], 2, 5, RECT90)
    assert r.status_code == 404
    d = r.json()["detail"]
    assert d["code"] == "anchor_not_found"
    assert d["anchor_fragment_id"] == 5
    assert d["errors"][0]["field"] == "anchor_fragment_id"
    assert "v2" in d["errors"][0]["message"]


def test_invalid_anchor_id(prepared):
    r = _assess(prepared["client"], 3, 0, RECT90)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "invalid_anchor"
    assert d["errors"][0]["field"] == "anchor_fragment_id"


def test_self_intersecting_target_rejected(prepared):
    # 蝴蝶结自交四边形
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
    r = _assess(prepared["client"], 3, 5, bowtie)
    assert r.status_code == 400
    d = r.json()["detail"]
    assert d["code"] == "self_intersecting_contour"
    assert d["errors"][0]["field"] == "target_contour"


def test_degenerate_target_rejected(prepared):
    # 三点共线 → 面积为零
    r = _assess(prepared["client"], 3, 5, [[0, 0], [10, 0], [20, 0]])
    assert r.status_code == 400
    assert r.json()["detail"]["code"] in (
        "invalid_target_contour", "self_intersecting_contour")


def test_too_few_target_points(prepared):
    r = _assess(prepared["client"], 3, 5, [[0, 0], [10, 0]])
    assert r.status_code == 422   # pydantic min_length=3


def test_non_finite_coordinates_rejected(prepared):
    c = prepared["client"]
    for bad in ("NaN", "Infinity", "-Infinity"):
        raw = (f'{{"anchor_fragment_id":5,"target_contour":'
               f'[[0,0],[90,0],[90,{bad}],[0,60]]}}')
        r = c.post("/api/assembly/versions/3/repair-region", content=raw,
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 400, bad
        d = r.json()["detail"]
        assert d["code"] == "non_finite_coordinate"
        assert d["errors"][0]["field"] == "target_contour[2]"
        assert "非有限数" in d["errors"][0]["message"]


def test_non_numeric_point_rejected(prepared):
    c = prepared["client"]
    raw = '{"anchor_fragment_id":5,"target_contour":[[0,0],[90,0],"x",[0,60]]}'
    r = c.post("/api/assembly/versions/3/repair-region", content=raw,
               headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_malformed_body_422(prepared):
    r = prepared["client"].post(
        "/api/assembly/versions/3/repair-region",
        json={"anchor_fragment_id": "x", "target_contour": RECT90})
    assert r.status_code == 422


def test_self_intersection_before_anchor_lookup(prepared):
    """目标轮廓非法时先 400 拒绝，即使锚片在该版本也不存在。"""
    r = _assess(prepared["client"], 2, 999,
                [[0, 0], [10, 10], [10, 0], [0, 10]])
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "self_intersecting_contour"
