bl_info = {
    "name": "Lip Sync Shapekeys",
    "author": "Mumulhl",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar",
    "description": "Use Rhubarb Lip Sync to keyframe shapekeys on the active mesh object",
    "category": "Animation",
}

import json
import math
import platform
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Context, Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper, ImportHelper


RHUBARB_KEYS = ("A", "B", "C", "D", "E", "F", "G", "H", "X")
RHUBARB_EXTENDED_SHAPES = "GHX"
SUPPORTED_AUDIO_EXTS = {"wav", "ogg"}


@dataclass
class MouthCue:
    value: str
    start: float
    end: float


def addon_dir() -> Path:
    return Path(__file__).resolve().parent


def executable_name() -> str:
    return "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"


def find_rhubarb_binary() -> Optional[Path]:
    candidate = addon_dir() / "bin" / executable_name()
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def abspath(path: str) -> Path:
    return Path(bpy.path.abspath(path)).expanduser()


def audio_needs_transcode(audio_path: Path) -> bool:
    return audio_path.suffix.lower().lstrip(".") not in SUPPORTED_AUDIO_EXTS


def transcode_audio_with_ffmpeg(audio_path: Path) -> tuple[Path, Optional[Path]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH. Please install ffmpeg or use wav/ogg input.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="lip_sync_shapekeys_"))
    tmp_audio = tmp_dir / f"{audio_path.stem}.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "48000",
        str(tmp_audio),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffmpeg conversion failed")
    return tmp_audio, tmp_dir


def prepare_audio_for_rhubarb(audio_path: Path) -> tuple[Path, Optional[Path]]:
    if not audio_needs_transcode(audio_path):
        return audio_path, None
    return transcode_audio_with_ffmpeg(audio_path)


def fps_to_frame(time_value: float, fps: int, fps_base: float = 1.0) -> float:
    return round(time_value * fps / fps_base, 7)


def frame_to_time(frame_value: float, fps: int, fps_base: float = 1.0) -> float:
    return frame_value * fps_base / fps


def frame_ceil(frame_value: float) -> int:
    return int(math.ceil(frame_value))


def frame_floor(frame_value: float) -> int:
    return int(math.floor(frame_value))


def snap_frame(frame_value: float) -> float:
    return float(int(math.floor(frame_value + 0.5)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def normalize_cues(cues: Iterable[MouthCue]) -> List[MouthCue]:
    ordered = sorted(
        (cue for cue in cues if cue.end > cue.start and cue.value),
        key=lambda cue: (cue.start, cue.end, cue.value),
    )
    normalized: List[MouthCue] = []
    for cue in ordered:
        if normalized and normalized[-1].value == cue.value and cue.start <= normalized[-1].end + 1e-6:
            normalized[-1].end = max(normalized[-1].end, cue.end)
            continue
        normalized.append(cue)
    return normalized


def cue_span_to_frames(
    cue: MouthCue,
    fps: int,
    fps_base: float,
    min_duration_frames: float,
    force_frame_intersection: bool,
) -> tuple[float, float]:
    start_frame = fps_to_frame(cue.start, fps, fps_base)
    end_frame = fps_to_frame(cue.end, fps, fps_base)

    if min_duration_frames > 0 and (end_frame - start_frame) < min_duration_frames:
        mid = (start_frame + end_frame) * 0.5
        half = min_duration_frames * 0.5
        start_frame = mid - half
        end_frame = mid + half

    if force_frame_intersection:
        start_right = frame_ceil(start_frame)
        end_left = frame_floor(end_frame)
        if end_left < start_right:
            start_left = frame_floor(start_frame)
            end_right = frame_ceil(end_frame)
            start_distance = start_frame - start_left
            end_distance = end_right - end_frame
            if start_distance < end_distance:
                start_frame = float(start_left)
            else:
                end_frame = float(end_right)

    return round(start_frame, 7), round(end_frame, 7)


def parse_rhubarb_output(stdout: str) -> List[MouthCue]:
    payload = json.loads(stdout)
    cues = payload.get("mouthCues", [])
    return [MouthCue(value=str(cue["value"]), start=float(cue["start"]), end=float(cue["end"])) for cue in cues]


def cue_list_from_scene(scene: bpy.types.Scene, cues: List[MouthCue]) -> List[MouthCue]:
    settings = scene.lss_settings
    processed = list(cues)
    if settings.merge_same_cues:
        processed = normalize_cues(processed)
    if settings.trim_long_cues:
        processed = trim_long_cues(processed, settings.long_cue_seconds)
    if settings.expand_short_cues:
        processed = expand_short_cues(processed, settings.short_cue_seconds)
    return processed


def trim_long_cues(cues: List[MouthCue], max_duration: float) -> List[MouthCue]:
    if max_duration <= 0:
        return cues
    trimmed: List[MouthCue] = []
    for cue in cues:
        if cue.end - cue.start > max_duration:
            cue = MouthCue(cue.value, cue.start, cue.start + max_duration)
        trimmed.append(cue)
    return trimmed


def expand_short_cues(cues: List[MouthCue], min_duration: float) -> List[MouthCue]:
    if min_duration <= 0:
        return cues
    expanded: List[MouthCue] = []
    for cue in cues:
        duration = cue.end - cue.start
        if duration < min_duration:
            mid = (cue.start + cue.end) * 0.5
            half = min_duration * 0.5
            cue = MouthCue(cue.value, mid - half, mid + half)
        expanded.append(cue)
    return expanded


def ensure_mapping_rows(scene: bpy.types.Scene) -> None:
    if not hasattr(scene, "lss_mouth_map"):
        return
    items = scene.lss_mouth_map
    if len(items) == 0:
        for key in RHUBARB_KEYS:
            item = items.add()
            item.rhubarb_key = key
            item.shape_key_name = ""
        return

    if len(items) < len(RHUBARB_KEYS):
        existing = {item.rhubarb_key for item in items}
        for key in RHUBARB_KEYS:
            if key in existing:
                continue
            item = items.add()
            item.rhubarb_key = key
            item.shape_key_name = ""
        return

    for index, key in enumerate(RHUBARB_KEYS):
        items[index].rhubarb_key = key


def mapping_dict(scene: bpy.types.Scene) -> Dict[str, str]:
    ensure_mapping_rows(scene)
    mapping: Dict[str, str] = {}
    for item in scene.lss_mouth_map:
        key = item.rhubarb_key.strip().upper()
        shape_name = item.shape_key_name.strip()
        if key:
            mapping[key] = shape_name
    return mapping


def clear_mapping(scene: bpy.types.Scene) -> None:
    ensure_mapping_rows(scene)
    for item in scene.lss_mouth_map:
        item.shape_key_name = ""


def serialize_mapping(scene: bpy.types.Scene) -> Dict[str, Any]:
    ensure_mapping_rows(scene)
    return {
        "version": 1,
        "items": [
            {
                "rhubarb_key": item.rhubarb_key,
                "shape_key_name": item.shape_key_name,
            }
            for item in scene.lss_mouth_map
        ],
    }


def deserialize_mapping(scene: bpy.types.Scene, payload: Any) -> None:
    ensure_mapping_rows(scene)
    clear_mapping(scene)

    items: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_items = payload.get("items", payload.get("mapping", []))
        if isinstance(raw_items, dict):
            items = [{"rhubarb_key": k, "shape_key_name": v} for k, v in raw_items.items()]
        elif isinstance(raw_items, list):
            items = [x for x in raw_items if isinstance(x, dict)]
    elif isinstance(payload, list):
        items = [x for x in payload if isinstance(x, dict)]

    by_key = {item.rhubarb_key.strip().upper(): item for item in scene.lss_mouth_map}
    for entry in items:
        key = str(entry.get("rhubarb_key", entry.get("key", ""))).strip().upper()
        shape_name = str(entry.get("shape_key_name", entry.get("shape", entry.get("value", "")))).strip()
        if key in by_key:
            by_key[key].shape_key_name = shape_name


def get_shape_key_block(obj: bpy.types.Object, shape_key_name: str):
    if not obj or obj.type != "MESH" or not obj.data or not obj.data.shape_keys:
        return None
    return obj.data.shape_keys.key_blocks.get(shape_key_name)


def ensure_action(shape_keys: bpy.types.Key, name: str) -> bpy.types.Action:
    if not shape_keys.animation_data:
        shape_keys.animation_data_create()
    action = shape_keys.animation_data.action
    if action is None:
        action = bpy.data.actions.new(name=name)
    elif not action.name:
        action.name = name
    if hasattr(action, "id_root"):
        action.id_root = "KEY"
    if hasattr(shape_keys.animation_data, "action_slot") and hasattr(action, "slots"):
        try:
            slot = None
            if len(action.slots) == 0:
                slot = action.slots.new(id_type="KEY", name=shape_keys.name or name)
            else:
                slot = next((s for s in action.slots if getattr(s, "target_id_type", "") == "KEY"), action.slots[0])
            shape_keys.animation_data.action_slot = slot
        except Exception:
            pass
    shape_keys.animation_data.action = action
    return action


def ensure_fcurve(action: bpy.types.Action, data_path: str) -> bpy.types.FCurve:
    for fcurve in action.fcurves:
        if fcurve.data_path == data_path and fcurve.array_index == 0:
            return fcurve
    return action.fcurves.new(data_path=data_path, index=0)


def clear_fcurve(action: bpy.types.Action, data_path: str) -> None:
    stale_fcurves = [fcurve for fcurve in action.fcurves if fcurve.data_path == data_path and fcurve.array_index == 0]
    for fcurve in stale_fcurves:
        action.fcurves.remove(fcurve)


def insert_keyframe(
    action: bpy.types.Action,
    data_path: str,
    frame: float,
    value: float,
    interpolation: str,
) -> None:
    fcurve = ensure_fcurve(action, data_path)
    for keyframe in fcurve.keyframe_points:
        if math.isclose(keyframe.co.x, frame, abs_tol=1e-4):
            keyframe.co.y = value
            keyframe.interpolation = interpolation
            if interpolation == "BEZIER":
                keyframe.handle_left_type = "AUTO_CLAMPED"
                keyframe.handle_right_type = "AUTO_CLAMPED"
            return
    keyframe = fcurve.keyframe_points.insert(frame, value)
    keyframe.interpolation = interpolation
    if interpolation == "BEZIER":
        keyframe.handle_left_type = "AUTO_CLAMPED"
        keyframe.handle_right_type = "AUTO_CLAMPED"


def cue_middle_start_frame(start_frame: float, end_frame: float, blend_inout_ratio: float) -> float:
    ratio = max(0.0, min(1.0, blend_inout_ratio))
    return start_frame * (1.0 - ratio) + end_frame * ratio


def cue_middle_end_frame(start_frame: float, end_frame: float, blend_inout_ratio: float) -> float:
    middle_start = cue_middle_start_frame(start_frame, end_frame, blend_inout_ratio)
    rounded = float(frame_ceil(middle_start))
    if rounded >= end_frame:
        return middle_start
    return rounded


def is_silence_cue(cue: MouthCue, use_extended_shapes: bool) -> bool:
    if cue.value == "X":
        return True
    if not use_extended_shapes and cue.value == "A":
        return True
    return False


def cue_center_frame(start_frame: float, end_frame: float) -> float:
    return (start_frame + end_frame) * 0.5


def cue_peak_value(
    cue: MouthCue,
    start_frame: float,
    end_frame: float,
    prev_cue: Optional[MouthCue],
    next_cue: Optional[MouthCue],
    prev_span: Optional[tuple[float, float]],
    next_span: Optional[tuple[float, float]],
    settings: "LSS_Settings",
) -> float:
    duration_frames = max(0.0, end_frame - start_frame)
    full_duration_frames = max(0.001, float(settings.peak_full_value_frames))
    duration_t = clamp01(duration_frames / full_duration_frames)
    duration_factor = lerp(float(settings.peak_min_value), 1.0, duration_t)

    center_frame = cue_center_frame(start_frame, end_frame)
    center_distances: List[float] = []
    if prev_cue and prev_span and not is_silence_cue(prev_cue, settings.use_extended_shapes):
        center_distances.append(center_frame - cue_center_frame(prev_span[0], prev_span[1]))
    if next_cue and next_span and not is_silence_cue(next_cue, settings.use_extended_shapes):
        center_distances.append(cue_center_frame(next_span[0], next_span[1]) - center_frame)

    if center_distances:
        nearest_center_distance = max(0.0, min(center_distances))
        full_neighbor_frames = max(0.001, float(settings.context_full_neighbor_frames))
        context_t = clamp01(nearest_center_distance / full_neighbor_frames)
        context_factor = lerp(float(settings.context_peak_floor), 1.0, context_t)
    else:
        context_factor = 1.0

    density_factor = 1.0
    if prev_cue and prev_span and next_cue and next_span:
        prev_gap = max(0.0, center_frame - cue_center_frame(prev_span[0], prev_span[1]))
        next_gap = max(0.0, cue_center_frame(next_span[0], next_span[1]) - center_frame)
        dense_gap = min(prev_gap, next_gap)
        dense_full_frames = max(0.001, float(settings.dense_context_full_frames))
        dense_t = clamp01(dense_gap / dense_full_frames)
        density_factor = lerp(float(settings.dense_context_floor), 1.0, dense_t)

    peak = duration_factor * context_factor * density_factor
    peak_floor = float(settings.peak_min_value) * float(settings.context_peak_floor)
    return max(peak_floor, min(1.0, peak))


def insert_cue_strip(
    action: bpy.types.Action,
    data_path: str,
    cue: MouthCue,
    prev_cue: Optional[MouthCue],
    next_cue: Optional[MouthCue],
    start_frame: float,
    end_frame: float,
    prev_span: Optional[tuple[float, float]],
    next_span: Optional[tuple[float, float]],
    settings: "LSS_Settings",
) -> None:
    interpolation = settings.keyframe_interpolation
    ratio = settings.blend_inout_ratio if settings.use_blend_ratio else 0.5
    ratio = max(0.0, min(1.0, ratio))
    peak_value = cue_peak_value(cue, start_frame, end_frame, prev_cue, next_cue, prev_span, next_span, settings)

    middle_start = cue_middle_start_frame(start_frame, end_frame, ratio)
    middle_end = cue_middle_end_frame(start_frame, end_frame, ratio)

    strip_start = start_frame
    if prev_cue and prev_span and not is_silence_cue(prev_cue, settings.use_extended_shapes):
        strip_start = cue_middle_end_frame(prev_span[0], prev_span[1], ratio)

    strip_end = end_frame
    if next_cue and next_span and not is_silence_cue(next_cue, settings.use_extended_shapes):
        strip_end = cue_middle_start_frame(next_span[0], next_span[1], ratio)

    strip_start = min(strip_start, middle_start)
    strip_end = max(strip_end, middle_end)
    if strip_end <= strip_start:
        strip_start = start_frame
        strip_end = end_frame
        middle_start = cue_middle_start_frame(start_frame, end_frame, ratio)
        middle_end = cue_middle_end_frame(start_frame, end_frame, ratio)

    if settings.force_frame_intersection:
        strip_start = snap_frame(strip_start)
        strip_end = snap_frame(strip_end)

        if middle_start < strip_start:
            middle_start = strip_start
        if middle_end < middle_start:
            middle_end = middle_start
        if strip_end <= middle_end:
            strip_end = middle_end + 1.0

    insert_keyframe(action, data_path, strip_start, 0.0, interpolation)
    if middle_start > strip_start + 1e-6:
        insert_keyframe(action, data_path, middle_start, peak_value, interpolation)
    else:
        insert_keyframe(action, data_path, strip_start, peak_value, interpolation)

    if middle_end > middle_start + 1e-6:
        insert_keyframe(action, data_path, middle_end, peak_value, interpolation)

    if strip_end > middle_end + 1e-6:
        insert_keyframe(action, data_path, strip_end, 0.0, interpolation)
    else:
        insert_keyframe(action, data_path, middle_end, 0.0, interpolation)


def apply_cues_to_mesh(scene: bpy.types.Scene, obj: bpy.types.Object, cues: List[MouthCue], mapping: Dict[str, str]) -> List[str]:
    shape_keys = obj.data.shape_keys
    assert shape_keys is not None
    if not cues:
        return ["No mouth cues were produced by Rhubarb."]

    settings = scene.lss_settings
    fps = scene.render.fps
    fps_base = scene.render.fps_base
    min_start = min(cue.start for cue in cues)
    max_end = max(cue.end for cue in cues)
    baseline_start = fps_to_frame(min_start, fps, fps_base) - float(settings.pre_roll_frames)
    baseline_start = max(float(scene.frame_start), baseline_start)
    baseline_end = fps_to_frame(max_end, fps, fps_base) + float(settings.post_roll_frames)

    action_name = f"LSS_{obj.name}"
    action = ensure_action(shape_keys, action_name)
    action.use_fake_user = True
    if hasattr(action, "frame_start"):
        action.frame_start = baseline_start
    if hasattr(action, "frame_end"):
        action.frame_end = baseline_end

    warnings: List[str] = []
    used_shape_keys = {name for name in mapping.values() if name}
    for shape_key_name in used_shape_keys:
        if get_shape_key_block(obj, shape_key_name) is None:
            warnings.append(f"Shape key '{shape_key_name}' was not found on '{obj.name}'.")
            continue
        clear_fcurve(action, f'key_blocks["{shape_key_name}"].value')

    # Put all mapped keys into a known zero state before the first cue.
    for shape_key_name in used_shape_keys:
        if get_shape_key_block(obj, shape_key_name) is None:
            continue
        insert_keyframe(
            action,
            f'key_blocks["{shape_key_name}"].value',
            baseline_start,
            0.0,
            settings.keyframe_interpolation,
        )

    cue_spans = [
        cue_span_to_frames(
            cue,
            fps,
            fps_base,
            settings.short_cue_seconds * fps / fps_base if settings.expand_short_cues else 0.0,
            settings.force_frame_intersection,
        )
        for cue in cues
    ]

    for index, cue in enumerate(cues):
        shape_key_name = mapping.get(cue.value)
        if not shape_key_name:
            warnings.append(f"No shapekey mapping for Rhubarb mouth shape '{cue.value}'.")
            continue

        if get_shape_key_block(obj, shape_key_name) is None:
            continue

        start_frame, end_frame = cue_spans[index]
        data_path = f'key_blocks["{shape_key_name}"].value'
        prev_cue = cues[index - 1] if index > 0 else None
        next_cue = cues[index + 1] if index + 1 < len(cues) else None
        prev_span = cue_spans[index - 1] if index > 0 else None
        next_span = cue_spans[index + 1] if index + 1 < len(cues) else None
        insert_cue_strip(
            action,
            data_path,
            cue,
            prev_cue,
            next_cue,
            start_frame,
            end_frame,
            prev_span,
            next_span,
            settings,
        )

    for shape_key_name in used_shape_keys:
        if get_shape_key_block(obj, shape_key_name) is None:
            continue
        insert_keyframe(
            action,
            f'key_blocks["{shape_key_name}"].value',
            baseline_end,
            0.0,
            settings.keyframe_interpolation,
        )
    shape_keys.animation_data.action = action
    return warnings


def rhubarb_command(audio_path: Path, binary: Path, use_extended_shapes: bool, recognizer: str) -> List[str]:
    args = [
        str(binary),
        "-f",
        "json",
        "--machineReadable",
    ]
    if use_extended_shapes:
        args.extend(["--extendedShapes", RHUBARB_EXTENDED_SHAPES])
    args.extend(["-r", recognizer, str(audio_path)])
    return args


class LSS_MouthMapItem(PropertyGroup):
    rhubarb_key: StringProperty(name="Rhubarb Key", default="")  # type: ignore
    shape_key_name: StringProperty(name="Shape Key", default="")  # type: ignore


class LSS_Settings(PropertyGroup):
    recognizer: EnumProperty(  # type: ignore
        name="识别器",
        items=[
            ("pocketSphinx", "pocketSphinx", "用于英语，通常效果最好"),
            ("phonetic", "phonetic", "用于非英语语音"),
        ],
        default="phonetic",
        description="选择 Rhubarb 的语音识别器。",
    )
    trim_long_cues: BoolProperty(  # type: ignore
        name="截断过长口型",
        default=False,
        description="过长的口型会被截断，避免拉太长。",
    )
    long_cue_seconds: FloatProperty(  # type: ignore
        name="最长口型秒数",
        default=0.35,
        min=0.01,
        max=10.0,
        description="超过这个时长的口型会被截断。",
    )
    expand_short_cues: BoolProperty(  # type: ignore
        name="扩展过短口型",
        default=True,
        description="过短口型会被扩展到最小可见时长。",
    )
    short_cue_seconds: FloatProperty(  # type: ignore
        name="最短口型秒数",
        default=0.12,
        min=0.0,
        max=1.0,
        description="短于这个时长的口型会被扩展。",
    )
    merge_same_cues: BoolProperty(  # type: ignore
        name="合并相邻同口型",
        default=True,
        description="合并连续重复的口型，减少重复关键帧。",
    )
    use_blend_ratio: BoolProperty(  # type: ignore
        name="使用进入/退出比例",
        default=True,
        description="将口型按进入/退出比例压缩到更明显的核心段。",
    )
    blend_inout_ratio: FloatProperty(  # type: ignore
        name="进入/退出比例",
        default=0.5,
        min=0.0,
        max=1.0,
        description="进入段和退出段的时长分配。0.5 表示两边相同。",
    )
    peak_min_value: FloatProperty(  # type: ignore
        name="最小峰值",
        default=0.35,
        min=0.0,
        max=1.0,
        description="极短口型或高密度切换时允许的最低峰值。",
    )
    peak_full_value_frames: FloatProperty(  # type: ignore
        name="满幅帧数",
        default=6.0,
        min=0.1,
        max=60.0,
        description="口型持续达到这个帧数后，不再因为时长不足而压低峰值。",
    )
    context_peak_floor: FloatProperty(  # type: ignore
        name="上下文下限",
        default=0.75,
        min=0.0,
        max=1.0,
        description="前后嘴型切换很密时，邻接上下文仍保留的最小峰值比例。",
    )
    context_full_neighbor_frames: FloatProperty(  # type: ignore
        name="邻接满幅帧距",
        default=7.0,
        min=0.1,
        max=60.0,
        description="与前后 cue 的中心距离达到这个帧数后，不再因为切换过密而压低峰值。",
    )
    dense_context_floor: FloatProperty(  # type: ignore
        name="高频下限",
        default=0.8,
        min=0.0,
        max=1.0,
        description="前后 cue 都很密时允许保留的最低峰值比例。",
    )
    dense_context_full_frames: FloatProperty(  # type: ignore
        name="高频满幅帧距",
        default=4.0,
        min=0.1,
        max=60.0,
        description="前后 cue 的中心距离达到这个帧数后，不再因为高频切换而继续压低峰值。",
    )
    pre_roll_frames: IntProperty(  # type: ignore
        name="前置缓冲帧",
        default=0,
        min=0,
        max=120,
        description="在第一段口型之前写入多少帧的归零关键帧",
    )
    post_roll_frames: IntProperty(  # type: ignore
        name="后置缓冲帧",
        default=1,
        min=0,
        max=120,
        description="在最后一段口型之后写入多少帧的归零关键帧",
    )
    force_frame_intersection: BoolProperty(  # type: ignore
        name="短口型对齐到帧",
        default=True,
        description="把短口型扩展到整数帧边界，让显示更稳定",
    )
    use_extended_shapes: BoolProperty(  # type: ignore
        name="使用扩展口型",
        default=True,
        description="把 Rhubarb 的扩展口型 G/H/X 也交给二进制处理",
    )
    keyframe_interpolation: EnumProperty(  # type: ignore
        name="关键帧插值",
        items=[
            ("BEZIER", "Bezier", "更平滑的嘴型起落"),
            ("LINEAR", "Linear", "更直接的线性切换"),
        ],
        default="BEZIER",
        description="控制嘴型关键帧的插值方式。",
    )


def draw_settings_block(layout: bpy.types.UILayout, settings: LSS_Settings) -> None:
    layout.use_property_split = True
    layout.use_property_decorate = False

    row = layout.row(align=True)
    row.prop(settings, "recognizer")
    layout.prop(settings, "use_extended_shapes")

    layout.separator()
    layout.label(text="口型整形")
    layout.prop(settings, "trim_long_cues")
    if settings.trim_long_cues:
        layout.prop(settings, "long_cue_seconds")
    layout.prop(settings, "expand_short_cues")
    if settings.expand_short_cues:
        layout.prop(settings, "short_cue_seconds")
    layout.prop(settings, "merge_same_cues")

    layout.separator()
    layout.label(text="时间设置")
    layout.prop(settings, "pre_roll_frames")
    layout.prop(settings, "post_roll_frames")
    layout.prop(settings, "force_frame_intersection")

    layout.separator()
    layout.label(text="曲线设置")
    layout.prop(settings, "keyframe_interpolation")
    layout.prop(settings, "use_blend_ratio")
    if settings.use_blend_ratio:
        layout.prop(settings, "blend_inout_ratio")
    layout.separator()
    layout.label(text="动态峰值")
    layout.prop(settings, "peak_min_value")
    layout.prop(settings, "peak_full_value_frames")
    layout.prop(settings, "context_peak_floor")
    layout.prop(settings, "context_full_neighbor_frames")
    layout.prop(settings, "dense_context_floor")
    layout.prop(settings, "dense_context_full_frames")


class LSS_OT_rhubarb_lip_sync(Operator):
    bl_idname = "lss.rhubarb_lip_sync"
    bl_label = "口型同步"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        obj = context.active_object
        return bool(obj and obj.type == "MESH")

    def execute(self, context: Context):
        scene = context.scene
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Please select a mesh object first.")
            return {"CANCELLED"}

        if not obj.data.shape_keys:
            self.report({"ERROR"}, "The selected mesh has no shape keys.")
            return {"CANCELLED"}

        audio_path = abspath(scene.lss_audio_path)
        if not audio_path.exists():
            self.report({"ERROR"}, f"Audio file does not exist: {audio_path}")
            return {"CANCELLED"}

        binary = find_rhubarb_binary()
        if binary is None:
            self.report({"ERROR"}, "Rhubarb binary was not found in the addon bin folder.")
            return {"CANCELLED"}

        temp_dir: Optional[Path] = None
        try:
            prepared_audio, temp_dir = prepare_audio_for_rhubarb(audio_path)
            if prepared_audio != audio_path:
                self.report({"INFO"}, f"Converted '{audio_path.name}' to a temporary WAV for Rhubarb.")

            cmd = rhubarb_command(prepared_audio, binary, scene.lss_settings.use_extended_shapes, scene.lss_settings.recognizer)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or f"Rhubarb exited with code {result.returncode}"
                self.report({"ERROR"}, message[:500])
                return {"CANCELLED"}

            try:
                cues = cue_list_from_scene(scene, parse_rhubarb_output(result.stdout))
            except Exception as exc:
                self.report({"ERROR"}, f"Failed to parse Rhubarb output: {exc}")
                return {"CANCELLED"}
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        mapping = mapping_dict(scene)
        warnings = apply_cues_to_mesh(scene, obj, cues, mapping)
        if warnings:
            for warning in warnings[:8]:
                self.report({"WARNING"}, warning)

        context.view_layer.update()
        self.report({"INFO"}, f"Applied {len(cues)} mouth cues to '{obj.name}'.")
        return {"FINISHED"}


class LSS_OT_save_mapping_preset(Operator, ExportHelper):
    bl_idname = "lss.save_mapping_preset"
    bl_label = "Save Mapping Preset"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore

    def execute(self, context: Context):
        scene = context.scene
        payload = serialize_mapping(scene)
        path = Path(self.filepath)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.report({"INFO"}, f"Saved mapping preset to {path}")
        return {"FINISHED"}


class LSS_OT_load_mapping_preset(Operator, ImportHelper):
    bl_idname = "lss.load_mapping_preset"
    bl_label = "Load Mapping Preset"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})  # type: ignore

    def execute(self, context: Context):
        scene = context.scene
        path = Path(self.filepath)
        payload = json.loads(path.read_text(encoding="utf-8"))
        deserialize_mapping(scene, payload)
        self.report({"INFO"}, f"Loaded mapping preset from {path}")
        return {"FINISHED"}


class LSS_OT_clear_mapping_preset(Operator):
    bl_idname = "lss.clear_mapping_preset"
    bl_label = "Clear Mapping"
    bl_options = {"UNDO"}

    def execute(self, context: Context):
        clear_mapping(context.scene)
        self.report({"INFO"}, "Cleared mapping values")
        return {"FINISHED"}


class LSS_PT_panel(Panel):
    bl_idname = "LSS_PT_panel"
    bl_label = "Lip Sync Shapekeys"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lip Sync"

    def draw(self, context: Context) -> None:
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        layout.prop(scene, "lss_audio_path", text="Audio File")

        settings_box = layout.box()
        settings_box.label(text="Rhubarb 参数")
        settings = scene.lss_settings
        draw_settings_block(settings_box, settings)

        box = layout.box()
        box.label(text="Rhubarb to Shape Key Map")
        row = box.row(align=True)
        row.operator(LSS_OT_save_mapping_preset.bl_idname, text="保存预设", icon="EXPORT")
        row.operator(LSS_OT_load_mapping_preset.bl_idname, text="导入预设", icon="IMPORT")
        row.operator(LSS_OT_clear_mapping_preset.bl_idname, text="清空", icon="TRASH")
        row = box.row(align=True)
        row.label(text="Rhubarb")
        row.label(text="Shape Key")
        for item in scene.lss_mouth_map:
            row = box.row(align=True)
            row.label(text=item.rhubarb_key)
            row.prop(item, "shape_key_name", text="")

        layout.separator()

        can_run = bool(obj and obj.type == "MESH")
        if not can_run:
            err_box = layout.box()
            err_box.alert = True
            err_box.label(text="Select a mesh object to enable lip sync.", icon="ERROR")

        op_row = layout.row()
        op_row.enabled = can_run
        op_row.operator(LSS_OT_rhubarb_lip_sync.bl_idname, text="口型同步")


classes = (
    LSS_MouthMapItem,
    LSS_Settings,
    LSS_OT_save_mapping_preset,
    LSS_OT_load_mapping_preset,
    LSS_OT_clear_mapping_preset,
    LSS_OT_rhubarb_lip_sync,
    LSS_PT_panel,
)


def init_scene(scene: bpy.types.Scene) -> None:
    ensure_mapping_rows(scene)


def init_all_scenes() -> Optional[float]:
    try:
        for scene in bpy.data.scenes:
            init_scene(scene)
    except Exception:
        return 0.5
    return None


@persistent
def depsgraph_update_handler(_depsgraph) -> None:
    for scene in bpy.data.scenes:
        init_scene(scene)


@persistent
def load_post_handler(_dummy) -> None:
    for scene in bpy.data.scenes:
        init_scene(scene)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lss_audio_path = StringProperty(
        name="Audio File",
        subtype="FILE_PATH",
        default="",
    )
    bpy.types.Scene.lss_mouth_map = CollectionProperty(type=LSS_MouthMapItem)
    bpy.types.Scene.lss_settings = bpy.props.PointerProperty(type=LSS_Settings)
    bpy.types.Scene.lss_mouth_map_index = IntProperty(default=0)
    bpy.app.timers.register(init_all_scenes, first_interval=0.1)
    if depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)
    if load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_post_handler)


def unregister() -> None:
    if depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)
    if load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_post_handler)
    del bpy.types.Scene.lss_mouth_map_index
    del bpy.types.Scene.lss_settings
    del bpy.types.Scene.lss_mouth_map
    del bpy.types.Scene.lss_audio_path
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
