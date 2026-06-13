# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm.model_executor.models.kimi_k25 import (
    _format_video_timestamp,
    _frames_to_pil_images,
    _split_video_chunks,
)

pytestmark = pytest.mark.skip_global_cleanup


class FakeImageProcessor:
    media_proc_cfg = {
        "temporal_merge_kernel_size": 2,
        "timestamp_mode": "hh:mm:ss.fff",
    }


def test_format_video_timestamp() -> None:
    assert _format_video_timestamp(0) == "00:00:00.000"
    assert _format_video_timestamp(65.25, "mm:ss.fff") == "01:05.250"
    assert _format_video_timestamp(65.25, "mm:ss") == "01:05"
    assert _format_video_timestamp(90061.5) == "25:01:01.500"
    assert _format_video_timestamp(-1) == "00:00:00.000"

    with pytest.raises(ValueError, match="Invalid Kimi video timestamp mode"):
        _format_video_timestamp(1, "seconds")


def test_frames_to_pil_images_accepts_thwc_and_tchw() -> None:
    thwc_frames = np.zeros((2, 5, 6, 3), dtype=np.uint8)
    tchw_frames = np.zeros((2, 3, 5, 6), dtype=np.float32)

    thwc_images = _frames_to_pil_images(thwc_frames)
    tchw_images = _frames_to_pil_images(tchw_frames)

    assert len(thwc_images) == 2
    assert len(tchw_images) == 2
    assert all(image.mode == "RGB" for image in thwc_images + tchw_images)
    assert thwc_images[0].size == (6, 5)
    assert tchw_images[0].size == (6, 5)


def test_frames_to_pil_images_accepts_pil_list() -> None:
    image = Image.new("RGBA", (2, 3))

    images = _frames_to_pil_images([image])

    assert len(images) == 1
    assert images[0].mode == "RGB"
    assert images[0].size == (2, 3)


def test_split_video_chunks_uses_video_metadata_for_timestamps() -> None:
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)

    chunks = _split_video_chunks(
        (
            frames,
            {
                "fps": 10.0,
                "frames_indices": [0, 10, 20],
            },
        ),
        FakeImageProcessor(),
    )

    assert len(chunks) == 2
    assert chunks[0]["type"] == "video_chunk"
    assert len(chunks[0]["video_chunk"]) == 2
    assert chunks[0]["prompt"].startswith("00:00:00.000")
    assert chunks[1]["prompt"].startswith("00:00:02.000")


def test_split_video_chunks_rejects_invalid_metadata() -> None:
    frames = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="frames_indices length"):
        _split_video_chunks(
            (frames, {"fps": 1.0, "frames_indices": [0]}),
            FakeImageProcessor(),
        )

    with pytest.raises(ValueError, match="fps must be positive"):
        _split_video_chunks((frames, {"fps": 0}), FakeImageProcessor())
