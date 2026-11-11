"""各 YOLO 数据集的类别 id → 英文名 / 中文名，以及宏观类别定义。

所有数据集共享一个「中文类名 → 英文 ref」的全局映射（CLASS_NAMES_ZH / EN_BY_ZH），
中文名在数据集之间必须唯一，因为 jsonl 里的 question / thinking 都是从中文类名反推英文的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_TRAIN_DATA = Path(__file__).resolve().parents[2] / "train_data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # 短名：data/images/<name> 软链 + jsonl image 前缀
    root: Path                      # YOLO 根目录（含 data.yaml / images / labels）
    classes_en: tuple[str, ...]     # 与 data.yaml names 顺序一致
    classes_zh: dict[str, str]      # en → zh
    has_split: bool                 # images/{train,val,test} 结构；否则按 hash 自动划分
    macro_zh: str | None = None     # 宏观类别中文名（如「船艇」），None 表示无宏观行
    macro_en: str | None = None
    macro_classes: frozenset[str] = field(default_factory=frozenset)  # 计入宏观行的细类


# ───────────────────────── ship_yolo ─────────────────────────
SHIP = DatasetSpec(
    name="ship",
    root=_TRAIN_DATA / "ship_yolo",
    classes_en=(
        "beacon",
        "cargo_ship",
        "coast_guard_ship",
        "cruise_ship",
        "fishing_boat",
        "oil_tanker_ship",
        "pennant_number",
        "sand_carrier_ship",
        "ship",
        "speedboat",
    ),
    classes_zh={
        "beacon": "航标",
        "cargo_ship": "货船",
        "coast_guard_ship": "海警船",
        "cruise_ship": "游轮",
        "fishing_boat": "渔船",
        "oil_tanker_ship": "油轮",
        "pennant_number": "舷号",
        "sand_carrier_ship": "砂石船",
        "ship": "船舶",
        "speedboat": "快艇",
    },
    has_split=True,
    macro_zh="船艇",
    macro_en="vessel",
    # 不含航标、舷号等非船体框
    macro_classes=frozenset(
        {
            "cargo_ship",
            "coast_guard_ship",
            "cruise_ship",
            "fishing_boat",
            "oil_tanker_ship",
            "sand_carrier_ship",
            "ship",
            "speedboat",
        }
    ),
)

# ───────────────────────── construction_vehicle_10cls ─────────────────────────
CONSTRUCTION = DatasetSpec(
    name="construction",
    root=_TRAIN_DATA / "construction_vehicle_10cls",
    classes_en=(
        "bulldozer",
        "construction_equipment",
        "crane",
        "excavator",
        "forklift",
        "mixer",
        "mixer_truck",
        "pile_driver",
        "road_roller",
        "tower_crane",
    ),
    classes_zh={
        "bulldozer": "推土机",
        "construction_equipment": "工程设备",
        "crane": "起重机",
        "excavator": "挖掘机",
        "forklift": "叉车",
        "mixer": "搅拌机",
        "mixer_truck": "搅拌车",
        "pile_driver": "打桩机",
        "road_roller": "压路机",
        "tower_crane": "塔吊",
    },
    has_split=False,
    macro_zh="工程车辆",
    macro_en="construction_vehicle",
    macro_classes=frozenset(
        {
            "bulldozer",
            "construction_equipment",
            "crane",
            "excavator",
            "forklift",
            "mixer",
            "mixer_truck",
            "pile_driver",
            "road_roller",
            "tower_crane",
        }
    ),
)

# ───────────────────────── mobile_vendor_mix ─────────────────────────
MOBILE_VENDOR = DatasetSpec(
    name="vendor",
    root=_TRAIN_DATA / "mobile_vendor_mix",
    classes_en=("mobile_vendor",),
    classes_zh={"mobile_vendor": "流动摊贩"},
    has_split=False,
)

# ───────────────────────── load_damage_yolo ─────────────────────────
ROAD_DAMAGE = DatasetSpec(
    name="road",
    root=_TRAIN_DATA / "load_damage_yolo",
    classes_en=(
        "road_crack",
        "road_repair",
        "road_aggregate_exposure",
        "road_collapse",
        "road_other_disease",
    ),
    classes_zh={
        "road_crack": "路面裂缝",
        "road_repair": "路面修补",
        "road_aggregate_exposure": "路面露骨料",
        "road_collapse": "路面塌陷",
        "road_other_disease": "路面其他病害",
    },
    has_split=True,
    macro_zh="路面病害",
    macro_en="road_damage",
    macro_classes=frozenset(
        {
            "road_crack",
            "road_repair",
            "road_aggregate_exposure",
            "road_collapse",
            "road_other_disease",
        }
    ),
)

ALL_DATASETS: tuple[DatasetSpec, ...] = (SHIP, CONSTRUCTION, MOBILE_VENDOR, ROAD_DAMAGE)
DATASETS_BY_NAME: dict[str, DatasetSpec] = {d.name: d for d in ALL_DATASETS}

# ───────────────────────── 全局合并映射 ─────────────────────────
# en → zh（细类 + 宏观类）
CLASS_NAMES_ZH: dict[str, str] = {}
for _d in ALL_DATASETS:
    CLASS_NAMES_ZH.update(_d.classes_zh)
    if _d.macro_zh and _d.macro_en:
        CLASS_NAMES_ZH[_d.macro_en] = _d.macro_zh

# zh → en
EN_BY_ZH: dict[str, str] = {v: k for k, v in CLASS_NAMES_ZH.items()}
assert len(EN_BY_ZH) == len(CLASS_NAMES_ZH), "中文类名在数据集之间发生冲突"

# 英文不规则复数（默认规则：以 s 结尾不变，否则 +s）
IRREGULAR_PLURAL_EN: dict[str, str] = {
    "construction_equipment": "construction_equipment",
}

# ───────────────────────── 向后兼容（ship 专用旧名） ─────────────────────────
CLASS_NAMES_EN: list[str] = list(SHIP.classes_en)
VESSEL_CLASSES: frozenset[str] = SHIP.macro_classes
MACRO_VESSEL_ZH = SHIP.macro_zh
