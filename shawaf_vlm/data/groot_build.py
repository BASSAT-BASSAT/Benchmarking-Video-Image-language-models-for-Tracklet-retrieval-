"""Build the GroOT-MOT17 tracklet mirror (one mp4 per captioned MOT17 track).

GroOT (Nguyen et al., NeurIPS 2023) attaches ``captions = [appearance, action]``
to every MOT17 box. This module crops each track from the MOT17 FRCNN frames,
writes one mp4 per track, and emits ``metadata/{all,appearance,action,combined}-test.jsonl``
in the same layout as bassatbassat/TVPReid. ``combined`` holds one refined sentence
per track that merges both GroOT captions (see ``groot_mot17_combined.json``).
"""

from __future__ import annotations

import configparser
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

GROOT_JSON_URL = (
    "https://raw.githubusercontent.com/uark-cviu/Type-to-Track/main/"
    "annotations/v1.0/mot17_train_coco.json"
)
MOT17_ZIP_URL = "https://motchallenge.net/data/MOT17.zip"
CONFIGS = ("all", "appearance", "action", "combined")
SUBSET = "groot_mot17"
SPLIT = "test"

_FRAME_RE = re.compile(r"(MOT17-\d+-[A-Z]+)_(\d+)\.jpg$")


@dataclass
class Track:
    sequence: str
    track_id: int
    appearance: str | None
    action: str | None
    boxes: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)

    @property
    def video_id(self) -> str:
        return f"{short_sequence(self.sequence)}_t{self.track_id:04d}"

    @property
    def video_rel(self) -> str:
        return f"videos/{self.video_id}.mp4"


def short_sequence(sequence: str) -> str:
    """``MOT17-02-FRCNN`` -> ``MOT17-02``."""

    parts = sequence.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else sequence


def clean_caption(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


SUBJECTS = ("man", "woman", "person", "boy", "girl", "child", "kid", "lady", "guy", "pedestrian")
GENERIC_SUBJECTS = ("person", "pedestrian")
_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")


def normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower()).rstrip(" .")
    return re.sub(r"\bt ?shirt\b", "t-shirt", text)


def split_subject(text: str) -> tuple[str | None, str]:
    """``"a young girl wearing pink"`` -> ``("young girl", "wearing pink")``.

    Up to two words may precede the subject noun.
    """

    body = _ARTICLE_RE.sub("", normalize_caption(text))
    words = body.split(" ")
    for index, word in enumerate(words[:3]):
        if word in SUBJECTS:
            rest = " ".join(words[index + 1 :])
            return " ".join(words[: index + 1]), re.sub(r"^(is|are)\s+", "", rest)
    return None, body


def combine_captions(appearance: str, action: str) -> str:
    """Merge GroOT's appearance and action captions into one sentence about one subject.

    The more specific subject wins (``woman`` over ``person``), so
    ``("person wearing black coat", "woman walking on the sidewalk")`` becomes
    ``"a woman wearing black coat, walking on the sidewalk"``.
    """

    look_subject, look = split_subject(appearance)
    act_subject, act = split_subject(action)
    specific = [s for s in (look_subject, act_subject) if s and s.split(" ")[-1] not in GENERIC_SUBJECTS]
    subject = specific[0] if specific else look_subject or act_subject or "person"
    if look and look_subject is None and not look.startswith(("wearing", "in ", "with ", "dressed", "carrying", "holding")):
        look = f"in {look}"
    article = "an" if subject[0] in "aeiou" else "a"
    parts = [f"{article} {subject}"]
    if look:
        parts.append(f" {look}")
    if act:
        parts.append(f", {act}" if look else f" {act}")
    return "".join(parts)


REFINED_CAPTIONS = Path(__file__).with_name("groot_mot17_combined.json")


def load_refined_captions(path: Path = REFINED_CAPTIONS) -> dict[str, str]:
    """LLM-refined ``combined`` captions keyed by ``video_id`` (empty if the file is missing)."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def combined_caption(track: Track, refined: dict[str, str]) -> str | None:
    if not (track.appearance and track.action):
        return None
    return refined.get(track.video_id) or combine_captions(track.appearance, track.action)


def collect_tracks(
    coco: dict,
    min_height: float = 50.0,
    min_visibility: float = 0.25,
    min_frames: int = 8,
    max_frames: int = 300,
    detector: str = "FRCNN",
) -> list[Track]:
    """Group GroOT boxes into captioned tracks, one copy of each MOT17 sequence."""

    frames: dict[int, tuple[str, int]] = {}
    for image in coco.get("images", []):
        match = _FRAME_RE.search(str(image.get("file_name", "")))
        if match is None:
            continue
        frames[int(image["id"])] = (match.group(1), int(match.group(2)))

    tracks: dict[tuple[str, int], Track] = {}
    for ann in coco.get("annotations", []):
        frame = frames.get(int(ann["image_id"]))
        if frame is None:
            continue
        sequence, frame_number = frame
        if detector and not sequence.endswith(f"-{detector}"):
            continue
        if int(ann.get("ignore", 0)) or int(ann.get("category_id", 1)) != 1:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if h < min_height or w <= 1:
            continue
        if float(ann.get("visibility", 1.0)) < min_visibility:
            continue
        key = (sequence, int(ann["track_id"]))
        track = tracks.get(key)
        if track is None:
            captions = list(ann.get("captions") or []) + [None, None]
            track = Track(
                sequence=sequence,
                track_id=int(ann["track_id"]),
                appearance=clean_caption(captions[0]),
                action=clean_caption(captions[1]),
            )
            tracks[key] = track
        track.boxes.append((frame_number, (x, y, w, h)))

    kept: list[Track] = []
    for track in tracks.values():
        if track.appearance is None and track.action is None:
            continue
        track.boxes.sort(key=lambda item: item[0])
        if len(track.boxes) < min_frames:
            continue
        if len(track.boxes) > max_frames:
            picks = np.linspace(0, len(track.boxes) - 1, num=max_frames).round().astype(int)
            track.boxes = [track.boxes[int(i)] for i in np.unique(picks)]
        kept.append(track)
    kept.sort(key=lambda t: (t.sequence, t.track_id))
    return kept


def metadata_rows(
    tracks: Iterable[Track],
    config: str,
    refined: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """One row per track; the gallery is identical across configs, captions differ."""

    if config not in CONFIGS:
        raise ValueError(f"Unknown GroOT config {config!r}. Use {', '.join(CONFIGS)}.")
    refined = load_refined_captions() if refined is None else refined
    rows: list[dict[str, object]] = []
    for track in tracks:
        pairs = [("appearance", track.appearance), ("action", track.action)]
        if config == "combined":
            pairs = [("combined", combined_caption(track, refined))]
        elif config != "all":
            pairs = [pair for pair in pairs if pair[0] == config]
        pairs = [(kind, text) for kind, text in pairs if text]
        rows.append(
            {
                "video_id": track.video_id,
                "video": track.video_rel,
                "captions": [text for _, text in pairs],
                "caption_types": [kind for kind, _ in pairs],
                "subset": SUBSET,
                "split": SPLIT,
                "sequence": short_sequence(track.sequence),
                "track_id": track.track_id,
                "num_frames": len(track.boxes),
            }
        )
    return rows


def write_metadata(tracks: list[Track], out_dir: Path) -> None:
    metadata = out_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        path = metadata / f"{config}-{SPLIT}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in metadata_rows(tracks, config):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def crop_geometry(track: Track, height: int = 256) -> tuple[float, int]:
    """Median width/height of a track, and the output width at ``height``."""

    aspect = statistics.median(w / h for _, (_, _, w, h) in track.boxes)
    width = max(16, int(round(height * aspect / 2.0)) * 2)
    return aspect, width


def expand_box(
    box: tuple[float, float, float, float],
    aspect: float,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    """Grow the box around its centre to ``aspect`` (w/h), then clip to the frame."""

    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    x0 = int(max(0, round(cx - w / 2.0)))
    y0 = int(max(0, round(cy - h / 2.0)))
    x1 = int(min(image_w, round(cx + w / 2.0)))
    y1 = int(min(image_h, round(cy + h / 2.0)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


class FrameSource:
    """Read MOT17 train frames from an extracted folder or straight from MOT17.zip."""

    def __init__(self, mot17: Path):
        self.path = Path(mot17)
        self._zip: zipfile.ZipFile | None = None
        self._names: dict[str, str] = {}
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.path)
            for name in self._zip.namelist():
                marker = name.find("train/")
                if marker != -1:
                    self._names[name[marker:]] = name
        else:
            train = self.path / "train"
            if not train.is_dir():
                nested = list(self.path.rglob("train"))
                train = nested[0] if nested else train
            self._train = train

    def _read_bytes(self, rel: str) -> bytes | None:
        if self._zip is not None:
            name = self._names.get(rel)
            return self._zip.read(name) if name else None
        path = self._train.parent / rel
        return path.read_bytes() if path.is_file() else None

    def frame(self, sequence: str, frame_number: int) -> np.ndarray | None:
        import cv2

        data = self._read_bytes(f"train/{sequence}/img1/{frame_number:06d}.jpg")
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    def fps(self, sequence: str, default: float = 30.0) -> float:
        data = self._read_bytes(f"train/{sequence}/seqinfo.ini")
        if data is None:
            return default
        parser = configparser.ConfigParser()
        parser.read_string(data.decode("utf-8", errors="ignore"))
        try:
            return float(parser["Sequence"]["frameRate"])
        except (KeyError, ValueError):
            return default


def write_track_videos(
    tracks: list[Track],
    source: FrameSource,
    out_dir: Path,
    height: int = 256,
    overwrite: bool = False,
) -> list[Track]:
    """Write one mp4 per track. Each sequence is decoded once, frame by frame."""

    import cv2

    videos = out_dir / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    by_sequence: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        by_sequence[track.sequence].append(track)

    written: list[Track] = []
    for sequence, seq_tracks in by_sequence.items():
        todo = [t for t in seq_tracks if overwrite or not (out_dir / t.video_rel).is_file()]
        written.extend(t for t in seq_tracks if t not in todo)
        if not todo:
            continue
        fps = source.fps(sequence)
        geometry = {id(t): crop_geometry(t, height) for t in todo}
        writers: dict[int, object] = {}
        counts: dict[int, int] = defaultdict(int)
        per_frame: dict[int, list[tuple[Track, tuple[float, float, float, float]]]] = defaultdict(list)
        for track in todo:
            for frame_number, box in track.boxes:
                per_frame[frame_number].append((track, box))
        try:
            for frame_number in tqdm(sorted(per_frame), desc=short_sequence(sequence), unit="frame"):
                image = source.frame(sequence, frame_number)
                if image is None:
                    continue
                image_h, image_w = image.shape[:2]
                for track, box in per_frame[frame_number]:
                    aspect, width = geometry[id(track)]
                    x0, y0, x1, y1 = expand_box(box, aspect, image_w, image_h)
                    crop = cv2.resize(image[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_AREA)
                    writer = writers.get(id(track))
                    if writer is None:
                        path = out_dir / track.video_rel
                        writer = cv2.VideoWriter(
                            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                        writers[id(track)] = writer
                    writer.write(crop)
                    counts[id(track)] += 1
        finally:
            for writer in writers.values():
                writer.release()
        for track in todo:
            if counts[id(track)] > 0:
                written.append(track)
    written.sort(key=lambda t: (t.sequence, t.track_id))
    return written


DATASET_CARD = """---
license: cc-by-nc-sa-3.0
task_categories:
- text-to-video
- zero-shot-classification
language:
- en
tags:
- person-retrieval
- text-to-video-retrieval
- tracklets
- mot17
- groot
pretty_name: GroOT-MOT17 tracklets
configs:
- config_name: all
  data_files: metadata/all-test.jsonl
- config_name: appearance
  data_files: metadata/appearance-test.jsonl
- config_name: action
  data_files: metadata/action-test.jsonl
- config_name: combined
  data_files: metadata/combined-test.jsonl
---

# GroOT-MOT17 tracklets

A text-to-tracklet retrieval benchmark for MOT17 pedestrians. Each captioned
MOT17 track is cropped into its own short mp4. A text query must retrieve the
matching track from a gallery of {n_tracks} tracks. The layout is the same as
[bassatbassat/TVPReid](https://huggingface.co/datasets/bassatbassat/TVPReid), so the
same evaluation code runs on both.

This is an **unofficial derivative**. The videos come from MOT17 and the
captions come from GroOT. The credit for both belongs to their authors (see
below). What this repository adds is listed under *Our contribution*.

## Configs

All configs share the same {n_tracks}-track gallery. Only the queries differ.

| Config | Queries | Unique query texts | What a query is |
| --- | ---: | ---: | --- |
| `appearance` | {n_app} | {u_app} | GroOT appearance caption, as released |
| `action` | {n_act} | {u_act} | GroOT action caption, as released |
| `all` | {n_all} | {u_all} | both of the above |
| `combined` | {n_comb} | {u_comb} | **one refined sentence per track describing appearance and action** (ours) |

Tracks without a caption of the requested type stay in the gallery as distractors.

**Read the action numbers with care.** GroOT action captions are mostly generic.
The most common one, "{top_act}", is shared by {top_act_n} tracks. Only
{u_act} of {n_act} action captions are unique. An action query often matches
several tracks equally well, yet exactly one counts as correct. The `combined`
config was made to reduce this ambiguity.

## Original work (credit)

- **Captions:** GroOT, from *Type-to-Track: Retrieve Any Object via Prompt-based
  Tracking* by Pha Nguyen, Kha Gia Quach, Kris Kitani, and Khoa Luu (NeurIPS 2023).
  GroOT gives every MOT17 box `captions = [appearance, action]`. We use
  `mot17_train_coco.json` from
  [uark-cviu/Type-to-Track](https://github.com/uark-cviu/Type-to-Track). The
  authors mark the MOT17 test captions as sub-optimal, so we do not use them.
- **Videos and boxes:** MOT17 / MOT16 by Anton Milan, Laura Leal-Taixé, Ian Reid,
  Stefan Roth, and Konrad Schindler ([motchallenge.net](https://motchallenge.net)).

The appearance and action captions in this repository are the GroOT captions
**unchanged**, apart from whitespace trimming and dropping empty or `None` values.

## Our contribution

1. **Tracklet videos.** Each GroOT track is cropped from the MOT17 FRCNN frames into
   one mp4, so the track itself can be retrieved instead of individual boxes.
2. **Retrieval splits.** The `appearance`, `action`, `all`, and `combined` query
   sets share one gallery, in the TVPReid format.
3. **Refined combined captions (`combined`).** For the {n_comb} tracks that have
   both captions, the two GroOT captions were merged into one fluent sentence
   about a single subject. It works in two passes:
   - A rule-based merge (`combine_captions` in the build code) joins the two
     captions and keeps the more specific subject ("woman" over "person").
   - An LLM assistant (Cursor agent, Auto model routing) then reviewed all
     {n_comb} merged captions one by one. It fixed grammar and articles, removed
     repeated phrases such as "carrying a bag, walking down the street carrying
     two bags", and resolved two gender conflicts between a track's appearance
     and action captions by looking at the frames (MOT17-05 tracks 54 and 59).
     **No new visual attributes were added.** Every refined caption uses only
     facts already present in the two GroOT captions.

   The refined captions are versioned in
   [`groot_mot17_combined.json`]({code_url}/blob/main/shawaf_vlm/data/groot_mot17_combined.json).
4. **Build code**, so the dataset can be regenerated from the original sources:
   [`scripts/build_groot_mot17.py`]({code_url}/blob/main/scripts/build_groot_mot17.py).

## Construction

- Only the FRCNN copy of each sequence is used, because DPM and SDP share the same frames.
- Boxes with `ignore=1`, height < {min_height}px, or visibility < {min_visibility} are dropped.
- Tracks need at least {min_frames} boxes and at least one caption. Long tracks
  are subsampled uniformly to at most {max_frames} boxes.
- Each box is grown to the track's median aspect ratio, so the extra width is
  real scene context rather than black padding. It is then resized to height
  {height} and written at the sequence frame rate.

## Layout

```text
videos/<MOT17-XX>_t<track>.mp4
metadata/all-test.jsonl
metadata/appearance-test.jsonl
metadata/action-test.jsonl
metadata/combined-test.jsonl
```

Each row has these fields: `video_id`, `video`, `captions`, `caption_types`,
`subset`, `split`, `sequence`, `track_id`, `num_frames`.

## Usage

```python
# pip install "shawaf-vlm @ git+{code_url}.git"
from shawaf_vlm.data import load_groot

splits = load_groot("combined")  # downloads the metadata and mp4s on first use
print(len(splits.query), len(splits.gallery))
```

Standard protocol: each caption is a query, and its own track is the only
positive. Report Rank-1/5/10, mAP, and median rank over the whole gallery.

## Intended use and limitations

- Zero-shot evaluation only. There is no train split, and GroOT's MOT17 test
  captions are not included.
- Captions describe a single pedestrian in a crowded street, mall, or hallway.
  Many people wear similar clothes, so some queries are still ambiguous even in
  `combined` (for example "a person in a black jacket walking on the sidewalk").
- The `combined` captions are machine-refined. Wording can differ from what a
  human annotator would write, but the content is limited to the source captions.

## License

GroOT annotations and MOT17 videos are released under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). This
dataset, including the refined captions, keeps that license: non-commercial use,
with attribution, share-alike.

## Citation

If you use this dataset, cite GroOT and MOT17. If you use the tracklet splits or
the `combined` captions, please also cite this repository.

```bibtex
@article{{nguyen2023type,
  title   = {{Type-to-Track: Retrieve Any Object via Prompt-based Tracking}},
  author  = {{Nguyen, Pha and Quach, Kha Gia and Kitani, Kris and Luu, Khoa}},
  journal = {{Advances in Neural Information Processing Systems}},
  year    = {{2023}}
}}
@article{{milan2016mot16,
  title   = {{MOT16: A Benchmark for Multi-Object Tracking}},
  author  = {{Milan, Anton and Leal-Taix{{\\'e}}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad}},
  journal = {{arXiv:1603.00831}},
  year    = {{2016}}
}}
@misc{{shawaf_groot_mot17,
  title        = {{GroOT-MOT17 tracklets: text-to-tracklet retrieval splits with refined combined captions}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note         = {{Derived from GroOT (Nguyen et al., 2023) and MOT17. Code: {code_url}}},
  year         = {{2026}}
}}
```
"""

CODE_URL = "https://github.com/BASSAT-BASSAT/Benchmarking-Video-Image-language-models-for-Tracklet-retrieval-"


def write_dataset_card(
    tracks: list[Track],
    out_dir: Path,
    repo_id: str = "bassatbassat/GroOT-MOT17",
    **settings: object,
) -> None:
    rows = {config: metadata_rows(tracks, config) for config in CONFIGS}
    texts = {config: [text for row in rows[config] for text in row["captions"]] for config in CONFIGS}
    top_act, top_act_n = Counter(text.lower() for text in texts["action"]).most_common(1)[0]
    card = DATASET_CARD.format(
        n_tracks=len(tracks),
        n_all=len(texts["all"]),
        n_app=len(texts["appearance"]),
        n_act=len(texts["action"]),
        n_comb=len(texts["combined"]),
        u_all=len({t.lower() for t in texts["all"]}),
        u_app=len({t.lower() for t in texts["appearance"]}),
        u_act=len({t.lower() for t in texts["action"]}),
        u_comb=len({t.lower() for t in texts["combined"]}),
        top_act=top_act,
        top_act_n=top_act_n,
        repo_id=repo_id,
        code_url=CODE_URL,
        **settings,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")
