import os
import sys
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def analyze_with_ai(video_path: str, player_height_m: Optional[float] = None) -> Dict[str, Any]:
    # Ensure project root (two levels up from this file) is on sys.path
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from ai.analyze_video_yolo import analyze_video  # type: ignore
        logger.info(f"Starting YOLOv8 analysis for: {video_path}" + (f" (player height: {player_height_m}m)" if player_height_m else ""))
        result = analyze_video(video_path, player_height_m=player_height_m)
        logger.info(f"YOLOv8 analysis complete: {len(result.get('key_frames', []))} keyframes")
        return result
    except ImportError as e:
        logger.error(f"Failed to import YOLOv8 analyzer: {e}")
        raise
    except Exception as e:
        logger.error(f"YOLOv8 analysis error: {type(e).__name__}: {e}", exc_info=True)
        raise


def analyze_frames(
    toss_image_path: str,
    contact_image_path: str,
    player_height_m: Optional[float] = None,
    toss_contact_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Run frame-based benchmark extraction (pose + ball on two images)."""
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from ai.analyze_video_yolo import analyze_frames as _analyze_frames  # type: ignore
        return _analyze_frames(
            toss_image_path,
            contact_image_path,
            player_height_m=player_height_m,
            toss_contact_time_s=toss_contact_time_s,
        )
    except ImportError as e:
        logger.error(f"Failed to import frame analyzer: {e}")
        raise
    except Exception as e:
        logger.error(f"Frame analysis error: {type(e).__name__}: {e}", exc_info=True)
        raise



