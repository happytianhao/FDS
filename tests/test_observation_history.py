"""Check the visual input boundary against persisted RGB, without a simulator or API."""

import io
import json

import numpy as np
import pytest
from PIL import Image
from test_decision_recovery import Environment, Planner, good

from fds.task_harness.examples import TrainingVideo
from fds.task_harness.observation import ObservationWindow, sample_every_third
from fds.task_harness.protocol import DecisionValidationError, TaskDecision, schema_for
from fds.task_harness.runtime import TaskHarness
from fds.types import Frame, Observation, StepResult


def frame(index, cue=False):
    return Frame(index, {
        "front": np.full((4, 5, 3), index % 256, dtype=np.uint8),
        "wrist": np.full((3, 4, 3), (index + 100) % 256, dtype=np.uint8),
    }, np.zeros(7), cue, {
        camera: {"K": np.eye(3).tolist(), "world_to_camera": np.eye(4).tolist()}
        for camera in ("front", "wrist")
    })


class VideoEnvironment(Environment):
    """A reset demo and several fresh sensor frames per environment step."""

    def reset(self, episode):
        self.next_frame = 12
        return Observation("Inspect video", [frame(i, i < 11) for i in range(12)])

    def step(self, action):
        self.steps += 1
        frames = [frame(i) for i in range(self.next_frame, self.next_frame + 4)]
        self.next_frame += 4
        return StepResult(Observation("Inspect video", frames), False, False, "ongoing")


def review(window_id, ids):
    return {"kind": "auxiliary", "tool_calls": [{
        "name": "review_frames", "arguments": {"window_id": window_id, "frame_ids": ids},
    }]}


def finish():
    return {"kind": "environment", "subgoal": "Stop", "tool_calls": [{
        "name": "finish", "arguments": {"reason": "Fixture complete"},
    }]}


def save(fid):
    return {"op": "save", "frame_id": fid, "description": "Reference", "reason": "Compare"}


@pytest.mark.parametrize("size,expected", [
    (0, []), (1, [0]), (2, [0, 1]), (3, [0, 2]), (4, [0, 3]),
    (7, [0, 3, 6]), (11, [0, 3, 6, 9, 10]),
])
def test_stride_includes_short_clips_and_deduplicates_final(size, expected):
    assert sample_every_third(list(range(size))) == expected


@pytest.mark.parametrize("mode", ["off", "full"])
def test_fresh_inputs_retention_and_historical_review_are_independent(tmp_path, mode):
    def respond(context, call):
        value = good(context)
        if call == 1:
            value["keyframe_changes"] = [save(0)]
            value["tool_batch"]["tool_calls"][0]["arguments"]["steps"] = 3
        elif call == 2:
            value["keyframe_changes"] = [{"op": "delete", "frame_id": 0}, save(23)]
            value["tool_batch"]["tool_calls"][0]["arguments"]["steps"] = 2
        elif call == 3:
            # Includes a never-delivered frame and a deleted keyframe, after two batches.
            value["tool_batch"] = review("initial", [0, 1])
        elif call == 4:
            value["keyframe_changes"] = [save(1)]
            value["tool_batch"] = review("batch_0001", [13, 14])
        elif call == 5:
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "check_image_correspondence", "arguments": {
                    "window_id": "batch_0001", "frame_id": 13,
                    "source_camera": "front", "target_camera": "wrist",
                    "source_pixel": [1, 1], "target_pixel": [1, 1],
                },
            }]}
        elif call == 6:
            value["tool_batch"] = review("batch_0002", [31])
        elif call == 7:
            value["keyframe_changes"] = [{"op": "delete", "frame_id": fid} for fid in (23, 1)]
            # Valid individual calls, but their aggregate exceeds the tool step budget.
            value["tool_batch"]["tool_calls"] = [
                {"name": "wait", "arguments": {"steps": 8}} for _ in range(2)
            ]
        else:
            value["tool_batch"] = finish()
        return value

    env, planner = VideoEnvironment(), Planner(respond)
    out = tmp_path / "episode"
    summary = TaskHarness(
        env, planner, max_calls=8, max_steps=20, max_tool_steps=8,
        options={"max_review_frames": 2, "demo_frames": 1, "initial_live_frames": 1},
        fds_options={"contract_mode": mode, "postcheck_enabled": False},
    ).run(0, out)
    assert summary["planner_calls"] == 8 and env.steps == 5
    assert summary["decision_recovery"]["rejected_decisions"] == 0
    expected = [
        [0, 3, 6, 9, 10, 11], [12, 15, 18, 21, 23, 0], [24, 27, 30, 31, 23],
        [0, 1, 23], [13, 14, 23, 1], [23, 1], [31, 23, 1], [],
    ]
    for context, images, ids in zip(planner.inputs, planner.images, expected, strict=True):
        labels = context["image_labels"]
        assert [(row["frame"], row["camera"]) for row in labels] == [
            (fid, camera) for camera in ("front", "wrist") for fid in ids
        ]
        # Check the actual attached pixels, not just claimed frame labels.
        for label, data in zip(labels, images, strict=True):
            with Image.open(io.BytesIO(data)) as image:
                offset = 100 if label["camera"] == "wrist" else 0
                assert np.all(np.asarray(image) == (label["frame"] + offset) % 256)
        if mode == "full":
            assert set(context["available_evidence_frames"]) == set(ids)
        encoded_schema = json.dumps(schema_for(context))
        assert "observation_steps" not in encoded_schema
    assert planner.inputs[1]["observation"]["sampling"]["new_frame_count"] == 12
    assert planner.inputs[1]["observation"]["sampling"]["final_frame"] == 23
    assert all(row["source"] == "history_review"
               for row in planner.inputs[4]["image_labels"] if row["frame"] in (13, 14))
    assert all(row["window_id"] == "batch_0001"
               for row in planner.inputs[4]["image_labels"] if row["frame"] in (13, 14))
    assert planner.inputs[5]["observation"]["frames"] == []
    assert planner.inputs[7]["last_tool_result"]["executed_steps"] == 0
    assert planner.inputs[7]["observation"]["sampling"]["final_frame"] is None
    # The complete source sequences survive finish and keyframe deletion.
    index = json.loads((out / "observations/index.json").read_text())
    assert [w["window_id"] for w in index["windows"]] == ["initial", "batch_0001", "batch_0002"]
    assert len(list((out / "observations/frames").glob("*.png"))) == 32 * 2
    rows = [json.loads(line) for line in (out / "observations/frames.jsonl").read_text().splitlines()]
    assert set(row["frame"] for row in rows) == set(range(32))
    for row in rows:
        for camera, path in row["images"].items():
            with Image.open(out / "observations" / path) as image:
                assert np.array_equal(np.asarray(image), frame(row["frame"]).images[camera])
    for call in range(8):
        assert (out / "audit" / f"call_{call:04d}" / "input.json").is_file()


def test_archive_never_grants_unseen_evidence_and_rejects_bad_reviews(tmp_path):
    archive = ObservationWindow(tmp_path)
    archive.begin("initial", "initial")
    for fid in range(5):
        archive.add(frame(fid))
    archive.begin("batch_0001", "execution")
    archive.add(frame(5))
    with pytest.raises(ValueError, match="not supplied"):
        archive.geometry_metadata(2, {5}, "initial")
    with pytest.raises(ValueError, match="Unknown"):
        archive.review("missing", [2], 2)
    with pytest.raises(ValueError, match="outside"):
        archive.review("batch_0001", [2], 2)
    with pytest.raises(ValueError, match="unique"):
        archive.review("initial", [2, 2], 2)
    with pytest.raises(ValueError, match="max_review_frames"):
        archive.review("initial", [0, 1, 2], 2)
    assert archive.review("initial", [], 2) == []
    assert [f.index for f in archive.review("initial", [2, 3], 2)] == [2, 3]
    calibration, sizes = archive.geometry_metadata(2, {2}, "initial")
    assert calibration == frame(2).calibration and sizes["wrist"] == [4, 3]
    archive.close()
    assert len(list((tmp_path / "observations/frames").glob("*.png"))) == 12


def test_correction_has_no_unretained_images_and_can_explicitly_review(tmp_path):
    def respond(context, call):
        value = good(context)
        if call == 1:
            value["tool_batch"]["observation_steps"] = [1]
        elif call == 2:
            value["tool_batch"] = review("initial", [1])
        else:
            value["tool_batch"] = finish()
        return value

    planner = Planner(respond)
    summary = TaskHarness(VideoEnvironment(), planner, max_calls=3).run(0, tmp_path / "episode")
    assert summary["decision_recovery"]["rejected_decisions"] == 1
    assert planner.images[1] == []
    assert planner.inputs[1]["available_evidence_frames"] == []
    assert [r["frame"] for r in planner.inputs[2]["image_labels"]] == [1, 1]
    assert planner.inputs[2]["available_evidence_frames"] == [1]
    # A historical ID in the catalog is not a valid new evidence reference.
    value = good(planner.inputs[2])
    value["fds_contract"]["required_evidence"] = [{
        "id": "old", "purpose": "Not currently visible", "status": "resolved", "evidence_refs": [0],
    }]
    with pytest.raises(DecisionValidationError):
        TaskDecision.parse(value, schema_for(planner.inputs[2]), delivered_frames=[1])


def test_full_training_video_is_archived_and_reviewable_after_acting(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as data:
        group = data.create_group("episode_0")
        group["setup/seed"] = 7
        group["setup/task"] = np.bytes_("fixture")
        group["setup/task_goal"] = np.bytes_("Watch the demonstration")
        for fid in range(8):
            step = group.create_group(f"timestep_{fid}")
            step["info/is_video_demo"] = True
            for camera, rgb in frame(fid).images.items():
                step[f"obs/{camera}_rgb"] = rgb

    def load(output):
        return TrainingVideo(source, tmp_path / "metadata.json", {"seed": 7},
                             "robomme", "fixture", 0, 1, output)

    def respond(context, call):
        value = good(context)
        if call == 1:
            # The complete copy must already exist before the first model call.
            source.unlink()
        elif call == 2:
            value["tool_batch"] = review("initial", ["training:0:1"])
        else:
            value["tool_batch"] = finish()
        return value

    planner, out = Planner(respond), tmp_path / "episode"
    summary = TaskHarness(Environment(done_after=10), planner, examples=load, max_calls=3).run(0, out)
    assert summary["planner_calls"] == 3 and summary["steps"] == 1
    assert not summary["decision_recovery"]["rejected_decisions"]
    assert [r["frame"] for r in planner.inputs[0]["image_labels"] if r["origin"] == "training_example"] == [
        f"training:0:{fid}" for _ in ("front", "wrist") for fid in (0, 3, 6, 7)
    ]
    assert [r["frame"] for r in planner.inputs[1]["image_labels"]] == [1]
    assert [r["frame"] for r in planner.inputs[2]["image_labels"]] == ["training:0:1"] * 2
    assert all(r["source"] == "history_review" for r in planner.inputs[2]["image_labels"])
    rows = json.loads((out / "training_example/frames.json").read_text())
    assert len(rows) == 8
    for fid, row in rows.items():
        for camera, path in row["images"].items():
            with Image.open(out / "training_example" / path) as image:
                assert np.array_equal(np.asarray(image), frame(int(fid.rsplit(":", 1)[1])).images[camera])
