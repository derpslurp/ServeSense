"""
Benchmark management for professional serve comparison
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

BENCHMARK_DIR = "benchmarks"
BENCHMARK_FILE = os.path.join(BENCHMARK_DIR, "benchmarks.json")

# Get absolute path to data files (backend/app/data/)
_current_dir = os.path.dirname(os.path.abspath(__file__))  # backend/app/utils
_app_dir = os.path.dirname(_current_dir)  # backend/app
PRO_BENCHMARKS_FILE = os.path.join(_app_dir, "data", "pro_benchmarks.json")
LITERATURE_BENCHMARKS_FILE = os.path.join(_app_dir, "data", "literature_benchmarks.json")


def ensure_benchmark_dir():
    """Ensure benchmark directory exists"""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def save_benchmark(
    name: str,
    metrics: Dict[str, Any],
    source_video: Optional[str] = None,
    player_height_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Save professional serve metrics as a benchmark.

    Args:
        name: Name of the benchmark (e.g., "Pro Player Serve")
        metrics: Dictionary with metric values (toss_contact_time_s, contact_height_m, etc.)
        source_video: Optional video ID or path used to create this benchmark
        player_height_m: Optional player height in meters (for height-normalized comparison)
    Returns:
        Saved benchmark data
    """
    ensure_benchmark_dir()
    benchmarks = load_all_benchmarks()
    benchmark = {
        "id": f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "name": name,
        "metrics": metrics,
        "source_video": source_video,
        "created_at": datetime.now().isoformat(),
    }
    if player_height_m is not None and player_height_m > 0:
        benchmark["player_height_m"] = round(player_height_m, 3)
    
    benchmarks.append(benchmark)
    
    with open(BENCHMARK_FILE, 'w', encoding='utf-8') as f:
        json.dump(benchmarks, f, indent=2)
    
    return benchmark


def load_all_benchmarks() -> List[Dict[str, Any]]:
    """Load all saved benchmarks"""
    ensure_benchmark_dir()
    
    if not os.path.exists(BENCHMARK_FILE):
        return []
    
    try:
        with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def get_benchmark(benchmark_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific benchmark by ID (searches both user-saved and professional benchmarks)"""
    # First check professional benchmarks
    pro_benchmarks = load_pro_benchmarks()
    pro_match = next((b for b in pro_benchmarks if b.get('id') == benchmark_id), None)
    if pro_match:
        return pro_match
    
    # Then check user-saved benchmarks
    user_benchmarks = load_all_benchmarks()
    return next((b for b in user_benchmarks if b.get('id') == benchmark_id), None)


def delete_benchmark(benchmark_id: str) -> bool:
    """Delete a benchmark"""
    benchmarks = load_all_benchmarks()
    filtered = [b for b in benchmarks if b['id'] != benchmark_id]
    
    if len(filtered) == len(benchmarks):
        return False  # Not found
    
    with open(BENCHMARK_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2)
    
    return True


def save_pro_benchmark(
    name: str,
    metrics: Dict[str, Any],
    serve_type: str,
    camera_angle: str,
    description: Optional[str] = None,
    source: Optional[str] = None,
    player_height_m: Optional[float] = None,
    images: Optional[Dict[str, str]] = None,
    benchmark_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one professional benchmark to pro_benchmarks.json.
    images: optional dict with keys 'toss' and 'contact' (URL paths, e.g. /uploads/benchmarks/xxx/toss.jpg)
    benchmark_id: optional custom id (used when creating from frames so directory matches)
    """
    benchmarks = load_pro_benchmarks()
    bid = benchmark_id or f"{serve_type}_{camera_angle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    entry = {
        "id": bid,
        "name": name,
        "description": description or f"{name} - {serve_type} {camera_angle}",
        "serve_type": serve_type,
        "camera_angle": camera_angle,
        "metrics": metrics,
        "source": source or "Created from frames",
        "level": "professional",
    }
    if player_height_m is not None and player_height_m > 0:
        entry["player_height_m"] = round(player_height_m, 3)
    if images and isinstance(images, dict):
        entry["images"] = {k: v for k, v in images.items() if k in ("toss", "contact")}
    benchmarks.append(entry)
    with open(PRO_BENCHMARKS_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    return entry


def load_literature_benchmarks() -> List[Dict[str, Any]]:
    """Load reference benchmarks from literature (same schema as pro_benchmarks)."""
    if not os.path.exists(LITERATURE_BENCHMARKS_FILE):
        return []
    try:
        with open(LITERATURE_BENCHMARKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def load_pro_benchmarks() -> List[Dict[str, Any]]:
    """Load built-in professional serve benchmarks (pro_benchmarks.json + literature_benchmarks.json)."""
    import logging
    logger = logging.getLogger(__name__)

    result: List[Dict[str, Any]] = []
    if os.path.exists(PRO_BENCHMARKS_FILE):
        try:
            with open(PRO_BENCHMARKS_FILE, "r", encoding="utf-8") as f:
                result = json.load(f)
            logger.info(f"Loaded {len(result)} pro benchmarks from {PRO_BENCHMARKS_FILE}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading pro benchmarks: {e}")
    else:
        logger.warning(f"Pro benchmarks file not found: {PRO_BENCHMARKS_FILE}")

    literature = load_literature_benchmarks()
    if literature:
        result = result + literature
        logger.info(f"Added {len(literature)} literature benchmarks")
    return result


def get_all_benchmarks(include_pro: bool = True) -> List[Dict[str, Any]]:
    """Get all benchmarks including professional ones"""
    all_benchmarks = []
    
    # Add pro benchmarks
    if include_pro:
        pro_benchmarks = load_pro_benchmarks()
        all_benchmarks.extend(pro_benchmarks)
    
    # Add user-saved benchmarks
    user_benchmarks = load_all_benchmarks()
    all_benchmarks.extend(user_benchmarks)
    
    return all_benchmarks


# Height metrics: compare as ratio of player height when both heights are known
HEIGHT_METRIC_KEYS = ["contact_height_m", "elbow_height_m", "toss_height_m"]


def _normalize_benchmark_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure contact (head to hand) >= elbow (head to elbow). Some literature entries had elbow > contact;
    with an extended arm the hand is further from head than the elbow, so we clamp for consistent display."""
    m = dict(metrics)
    ch = m.get("contact_height_m")
    eh = m.get("elbow_height_m")
    if ch is not None and eh is not None and eh > ch:
        m["elbow_height_m"] = ch  # clamp so pro display matches user data (contact >= elbow)
    return m


def _get_ratio(value: Optional[float], height_m: Optional[float]) -> Optional[float]:
    """Return value/height_m if both are valid and height > 0."""
    if value is None or height_m is None or not (height_m > 0):
        return None
    try:
        return float(value) / float(height_m)
    except (TypeError, ZeroDivisionError):
        return None


def compare_with_benchmark(
    user_metrics: Dict[str, Any],
    benchmark_metrics: Dict[str, Any],
    user_height_m: Optional[float] = None,
    benchmark_height_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare user metrics against benchmark. When both user and benchmark have
    player height, height metrics (contact, elbow, toss) are compared as
    ratios (value / height) so different body sizes are comparable.
    """
    comparison: Dict[str, Any] = {
        "user": user_metrics,
        "benchmark": benchmark_metrics,
        "differences": {},
        "ratings": {},
        "overall_score": 0.0,
    }
    use_ratios = (
        user_height_m is not None
        and user_height_m > 0
        and benchmark_height_m is not None
        and benchmark_height_m > 0
    )
    if use_ratios:
        comparison["user_ratios"] = {}
        comparison["benchmark_ratios"] = {}

    metric_keys = ["toss_contact_time_s", "contact_height_m", "elbow_height_m", "toss_height_m"]
    scores = []

    for key in metric_keys:
        user_val = user_metrics.get(key)
        bench_val = benchmark_metrics.get(key)
        # Include metric when user has value; if benchmark missing, still show user data
        if user_val is None:
            continue
        if bench_val is None:
            comparison["differences"][key] = {"benchmark_missing": True}
            comparison["ratings"][key] = "no_pro_data"
            continue

        is_height_metric = key in HEIGHT_METRIC_KEYS
        if is_height_metric and use_ratios:
            user_ratio = _get_ratio(user_val, user_height_m)
            bench_ratio = _get_ratio(bench_val, benchmark_height_m)
            if user_ratio is not None and bench_ratio is not None:
                comparison["user_ratios"][key] = round(user_ratio, 4)
                comparison["benchmark_ratios"][key] = round(bench_ratio, 4)
                diff = user_ratio - bench_ratio
                percent_diff = abs(diff / bench_ratio * 100) if bench_ratio != 0 else 0
                comparison["differences"][key] = {
                    "absolute": diff,
                    "percent": percent_diff,
                    "compared_as": "ratio_of_height",
                }
            else:
                diff = float(user_val) - float(bench_val)
                percent_diff = abs(diff / float(bench_val) * 100) if bench_val != 0 else 0
                comparison["differences"][key] = {"absolute": diff, "percent": percent_diff}
        else:
            diff = float(user_val) - float(bench_val)
            percent_diff = abs(diff / float(bench_val) * 100) if bench_val != 0 else 0
            comparison["differences"][key] = {"absolute": diff, "percent": percent_diff}

        percent_diff = comparison["differences"][key].get("percent", 0)
        if percent_diff < 12:
            comparison["ratings"][key] = "excellent"
            scores.append(100)
        elif percent_diff < 25:
            comparison["ratings"][key] = "good"
            scores.append(75)
        elif percent_diff < 45:
            comparison["ratings"][key] = "needs_improvement"
            scores.append(50)
        else:
            comparison["ratings"][key] = "far_off"
            scores.append(25)

    if scores:
        comparison["overall_score"] = sum(scores) / len(scores)
    logger.debug(f"Comparison (use_ratios={use_ratios}): overall={comparison['overall_score']}")
    return comparison


def find_best_match(user_metrics: Dict[str, Any], serve_type: Optional[str] = None, camera_angle: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatically compare user metrics against literature benchmarks only
    and return the best match with comparison data
    
    Args:
        user_metrics: Dictionary of user's serve metrics
        serve_type: Optional serve type filter (jump_float_serve, standing_float_serve, jump_topspin_serve)
        camera_angle: Optional camera angle filter (front, side, diagonal_front, diagonal_side)
    
    Returns:
        Dictionary with best_match benchmark, comparison, and all comparisons
    """
    pro_benchmarks = load_literature_benchmarks()
    
    if not pro_benchmarks:
        return {
            "best_match": None,
            "comparison": None,
            "all_comparisons": [],
            "summary": {
                "total_compared": 0,
                "best_match_name": None,
                "best_match_score": 0.0,
            },
        }
    
    # Filter benchmarks by serve_type and camera_angle if provided
    filtered_benchmarks = pro_benchmarks
    if serve_type or camera_angle:
        filtered_benchmarks = []
        for benchmark in pro_benchmarks:
            benchmark_serve_type = benchmark.get("serve_type")
            benchmark_camera_angle = benchmark.get("camera_angle")
            
            # Match serve_type if provided
            serve_match = not serve_type or benchmark_serve_type == serve_type
            # Match camera_angle if provided
            angle_match = not camera_angle or benchmark_camera_angle == camera_angle
            
            if serve_match and angle_match:
                filtered_benchmarks.append(benchmark)
    
    if not filtered_benchmarks:
        # If no exact match, fall back to all benchmarks
        logger.warning(f"No benchmarks found matching serve_type={serve_type}, camera_angle={camera_angle}. Using all benchmarks.")
        filtered_benchmarks = pro_benchmarks
    
    # Official pro metrics for Detailed Metrics & Pro Comparison
    # Toss→contact: commonly cited 0.8–1.2 s (toss release to hit); literature sometimes reports ~0.5 (e.g. apex-to-contact or specific studies)
    PRO_ELBOW_M = 7 / 39.3701
    PRO_TOSS_CONTACT_S = 1.0  # typical pro range 0.8–1.2 s

    comparisons = []
    for benchmark in filtered_benchmarks:
        bench_metrics = dict(benchmark["metrics"])
        if "Official" in benchmark.get("name", ""):
            bench_metrics["elbow_height_m"] = PRO_ELBOW_M
            bench_metrics["toss_contact_time_s"] = PRO_TOSS_CONTACT_S
        bench_metrics = _normalize_benchmark_metrics(bench_metrics)  # contact >= elbow to match user data
        comparison = compare_with_benchmark(user_metrics, bench_metrics)
        comparisons.append({
            "benchmark": {**benchmark, "metrics": bench_metrics},
            "comparison": comparison,
            "overall_score": comparison["overall_score"]
        })
    
    # Sort by overall score (best match first)
    comparisons.sort(key=lambda x: x["overall_score"], reverse=True)
    
    best_match = comparisons[0] if comparisons else None
    
    # Log comparison results for debugging
    if best_match:
        logger.info(f"Best match: {best_match['benchmark']['name']} with score {best_match['overall_score']:.2f}%")
        logger.debug(f"User metrics: {user_metrics}")
        logger.debug(f"Best match metrics: {best_match['benchmark']['metrics']}")
        logger.debug(f"Individual ratings: {best_match['comparison']['ratings']}")
    
    return {
        "best_match": best_match,
        "all_comparisons": comparisons,
        "summary": {
            "total_compared": len(comparisons),
            "best_match_name": best_match["benchmark"]["name"] if best_match else None,
            "best_match_score": round(best_match["overall_score"], 2) if best_match else 0.0
        }
    }

