"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import axios from "axios";
import { useParams, useRouter } from "next/navigation";
import ScrollReveal from "../../../components/ScrollReveal";

interface KeyFrame {
  frame_index: number;
  image_path: string;
  label?: string;
}
interface AnalysisResponse {
  video_id?: string;
  duration_sec?: number;
  frame_count?: number;
  total_frames?: number;
  frames_analyzed?: number;
  fps?: number;
  camera_angle?: string;
  serve_type?: string;
  annotations?: {
    frame_index: number;
    preview_image?: string;
    image_path?: string;
  }[];
  key_moments?: Record<string, number>;
  key_frames?: KeyFrame[];
  feedback?: {
    tips: string[];
    recommendations?: string[];
    metrics?: {
      toss_contact_time_s?: number;
      contact_height_m?: number;
      elbow_height_m?: number;
      toss_height_m?: number;
    };
    output_dir?: string;
    pose_sample?: any[];
  };
  pro_comparison?: {
    best_match?: {
      benchmark: {
        name: string;
        description?: string;
        id: string;
        images?: { toss?: string; contact?: string };
      };
      comparison: {
        user: any;
        benchmark: any;
        differences: Record<string, { absolute: number; percent: number }>;
        ratings: Record<string, string>;
        overall_score: number;
      };
      overall_score: number;
    };
    summary?: {
      total_compared: number;
      best_match_name: string;
      best_match_score: number;
    };
  };
}

const labelColor: Record<string, string> = {
  preparation: "bg-blue-500",
  "Toss height": "bg-amber-500",
  toss: "bg-amber-500",
  jump: "bg-emerald-500",
  hit: "bg-sky-500",
  follow_through: "bg-fuchsia-500",
};

const METERS_TO_INCHES = 39.3701;

const formatInches = (meters?: number, decimals = 1) => {
  if (meters === undefined || meters === null || Number.isNaN(meters))
    return "—";
  const inches = meters * METERS_TO_INCHES;
  return `${inches.toFixed(decimals)} in`;
};

const formatSeconds = (seconds?: number, decimals = 2) => {
  if (seconds === undefined || seconds === null || Number.isNaN(seconds))
    return "—";
  return `${seconds.toFixed(decimals)} s`;
};

type MetricStatus = "excellent" | "good" | "attention" | "poor";

interface NumericMetricConfig {
  label: string;
  unit?: string;
  thresholds: { excellent: number; good: number; attention: number };
  toleranceType?: "absolute" | "percent";
  precision?: number;
  preferHigher?: boolean;
  preferLower?: boolean;
}

interface MetricEvaluation {
  status: MetricStatus;
  className: string;
  icon: string;
  statusLabel: string;
  detail?: string;
}

const statusStyles: Record<
  MetricStatus,
  { className: string; icon: string; statusLabel: string }
> = {
  excellent: {
    className: "text-emerald-600 dark:text-emerald-400",
    icon: "✅",
    statusLabel: "Match",
  },
  good: {
    className: "text-blue-600 dark:text-blue-400",
    icon: "✅",
    statusLabel: "Close",
  },
  attention: {
    className: "text-amber-600 dark:text-amber-400",
    icon: "⚠️",
    statusLabel: "Needs work",
  },
  poor: {
    className: "text-red-600 dark:text-red-400",
    icon: "❌",
    statusLabel: "Far off",
  },
};

// Only the 4 metrics we analyze (toss→contact time, contact height, elbow height, toss height)
const numericMetricConfigs: Record<string, NumericMetricConfig> = {
  toss_contact_time_s: {
    label: "Toss → Contact Time",
    unit: "s",
    thresholds: { excellent: 0.18, good: 0.35, attention: 0.55 },
    precision: 2,
  },
  contact_height_m: {
    label: "Contact (head to hand)",
    unit: "in",
    thresholds: { excellent: 0.1, good: 0.2, attention: 0.35 },
    precision: 2,
    preferHigher: true,
  },
  elbow_height_m: {
    label: "Elbow (head to elbow)",
    unit: "in",
    thresholds: { excellent: 0.1, good: 0.2, attention: 0.35 },
    precision: 2,
    preferHigher: true,
  },
  toss_height_m: {
    label: "Toss Height",
    unit: "in",
    thresholds: { excellent: 0.2, good: 0.4, attention: 0.6 },
    precision: 2,
  },
};

function evaluateNumericMetric(
  value: number | undefined,
  benchmark: number | undefined,
  config: NumericMetricConfig
): MetricEvaluation | null {
  if (value === undefined || value === null || Number.isNaN(value)) return null;
  if (benchmark === undefined || benchmark === null || Number.isNaN(benchmark))
    return null;

  const rawDiff = value - benchmark;
  const basePercentDiff =
    benchmark === 0 ? 0 : (rawDiff / Math.abs(benchmark)) * 100;
  const prefersDirection =
    (config.preferHigher && rawDiff >= 0) ||
    (config.preferLower && rawDiff <= 0);

  let comparisonDiff =
    config.toleranceType === "percent"
      ? Math.abs(basePercentDiff)
      : Math.abs(rawDiff);

  if (prefersDirection) {
    comparisonDiff = 0;
  }

  const { thresholds } = config;
  let status: MetricStatus;
  if (comparisonDiff <= thresholds.excellent) status = "excellent";
  else if (comparisonDiff <= thresholds.good) status = "good";
  else if (comparisonDiff <= thresholds.attention) status = "attention";
  else status = "poor";

  const style = statusStyles[status];
  const decimalPlaces =
    config.precision ??
    (config.unit === "°"
      ? 1
      : config.unit === "mph"
      ? 1
      : config.unit === "s"
      ? 2
      : 2);

  const diffVal =
    config.unit === "in" ? Math.abs(rawDiff) * METERS_TO_INCHES : Math.abs(rawDiff);
  const diffDisplay =
    config.toleranceType === "percent"
      ? `${Math.abs(basePercentDiff).toFixed(1)}%`
      : `${diffVal.toFixed(decimalPlaces)}${
          config.unit ? ` ${config.unit}` : ""
        }`;

  const sign = rawDiff >= 0 ? "+" : "-";
  const detail =
    Math.abs(rawDiff) < 1e-6
      ? "matches pro"
      : `${sign}${diffDisplay} ${rawDiff >= 0 ? "above" : "below"} pro${
          prefersDirection ? " (on target)" : ""
        }`;

  return {
    status,
    className: style.className,
    icon: style.icon,
    statusLabel: style.statusLabel,
    detail,
  };
}

function evaluateStanceClassification(
  userValue?: string,
  benchmarkValue?: string
): MetricEvaluation | null {
  if (!userValue || !benchmarkValue) return null;
  if (userValue === benchmarkValue) {
    const style = statusStyles.excellent;
    return {
      status: "excellent",
      className: style.className,
      icon: style.icon,
      statusLabel: "Match",
      detail: userValue,
    };
  }
  const style = statusStyles.attention;
  return {
    status: "attention",
    className: style.className,
    icon: style.icon,
    statusLabel: "Different",
    detail: `yours: ${userValue}, pro: ${benchmarkValue}`,
  };
}

const getNumericValue = (
  metrics: Record<string, unknown> | undefined,
  key: string
): number | undefined => {
  if (!metrics) return undefined;
  const value = metrics[key];
  if (typeof value === "number") return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

export default function ResultsPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = params?.id;
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [showBenchmarkModal, setShowBenchmarkModal] = useState(false);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [savingBenchmark, setSavingBenchmark] = useState(false);
  const [benchmarks, setBenchmarks] = useState<any[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(
    null
  );
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  // Upload another video states
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [quickServeType, setQuickServeType] = useState<string>("");
  const [quickCameraAngle, setQuickCameraAngle] = useState<string>("");

  const userMetrics = data?.feedback?.metrics;

  useEffect(() => {
    async function run() {
      if (!id) return;
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get(`/api/analyze/${id}`);
        console.log("Analysis response:", res.data);
        console.log("Pro comparison:", res.data.pro_comparison);
        setData(res.data);
      } catch (e: any) {
        // Handle FastAPI validation errors (detail can be an array or object)
        let errorMsg = "Failed to load analysis";
        if (e?.response?.data?.detail) {
          const detail = e.response.data.detail;
          if (Array.isArray(detail)) {
            errorMsg = detail
              .map((d: any) => d.msg || JSON.stringify(d))
              .join(", ");
          } else if (typeof detail === "string") {
            errorMsg = detail;
          } else {
            errorMsg = JSON.stringify(detail);
          }
        } else if (e?.response?.data?.error) {
          errorMsg =
            typeof e.response.data.error === "string"
              ? e.response.data.error
              : JSON.stringify(e.response.data.error);
        } else if (e?.message) {
          errorMsg = e.message;
        }
        setError(errorMsg);
      } finally {
        setLoading(false);
      }
    }
    run();
  }, [id]);

  useEffect(() => {
    if (data?.serve_type && !quickServeType) {
      setQuickServeType(data.serve_type);
    }
    if (data?.camera_angle && !quickCameraAngle) {
      setQuickCameraAngle(data.camera_angle);
    }
  }, [data?.serve_type, data?.camera_angle, quickServeType, quickCameraAngle]);

  // Close modal on ESC key
  useEffect(() => {
    function handleEscape(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setSelectedImage(null);
        setShowBenchmarkModal(false);
      }
    }
    if (selectedImage || showBenchmarkModal) {
      window.addEventListener("keydown", handleEscape);
      return () => window.removeEventListener("keydown", handleEscape);
    }
  }, [selectedImage, showBenchmarkModal]);

  // Load benchmarks
  useEffect(() => {
    async function loadBenchmarks() {
      try {
        const res = await axios.get("/api/benchmarks");
        setBenchmarks(res.data);
      } catch (e) {
        console.error("Failed to load benchmarks:", e);
      }
    }
    loadBenchmarks();
  }, []);

  async function saveAsBenchmark() {
    if (!benchmarkName.trim() || !id) return;
    const metricsPayload = data?.feedback?.metrics;
    if (!metricsPayload) {
      alert("No metrics available to save for this benchmark.");
      return;
    }
    setSavingBenchmark(true);
    try {
      await axios.post("/api/benchmarks", {
        name: benchmarkName,
        video_id: id,
        metrics: metricsPayload,
      });
      setShowBenchmarkModal(false);
      setBenchmarkName("");
      // Reload benchmarks
      const res = await axios.get("/api/benchmarks");
      setBenchmarks(res.data);
      alert("Benchmark saved successfully!");
    } catch (e: any) {
      // Handle FastAPI validation errors (detail can be an array or object)
      let errorMsg = "Failed to save benchmark";
      if (e?.response?.data?.detail) {
        const detail = e.response.data.detail;
        if (Array.isArray(detail)) {
          errorMsg = detail
            .map((d: any) => d.msg || JSON.stringify(d))
            .join(", ");
        } else if (typeof detail === "string") {
          errorMsg = detail;
        } else {
          errorMsg = JSON.stringify(detail);
        }
      } else if (e?.response?.data?.error) {
        errorMsg =
          typeof e.response.data.error === "string"
            ? e.response.data.error
            : JSON.stringify(e.response.data.error);
      } else if (e?.message) {
        errorMsg = e.message;
      }
      alert(`Failed to save benchmark: ${errorMsg}`);
    } finally {
      setSavingBenchmark(false);
    }
  }

  async function compareWithBenchmark() {
    if (!selectedBenchmark || !id) return;
    if (!data?.feedback?.metrics) {
      alert("No metrics available. Please wait for the analysis to complete.");
      return;
    }
    setLoadingComparison(true);
    setComparisonData(null);
    try {
      // Pass existing metrics from the analysis to avoid re-analyzing (video may be deleted)
      const res = await axios.post(
        `/api/compare/${id}/benchmark/${selectedBenchmark}`,
        {
          metrics: data.feedback.metrics,
        }
      );
      setComparisonData(res.data);
    } catch (e: any) {
      // Handle FastAPI validation errors (detail can be an array or object)
      let errorMsg = "Failed to compare";
      if (e?.response?.data?.detail) {
        const detail = e.response.data.detail;
        if (Array.isArray(detail)) {
          errorMsg = detail
            .map((d: any) => d.msg || JSON.stringify(d))
            .join(", ");
        } else if (typeof detail === "string") {
          errorMsg = detail;
        } else {
          errorMsg = JSON.stringify(detail);
        }
      } else if (e?.response?.data?.error) {
        errorMsg =
          typeof e.response.data.error === "string"
            ? e.response.data.error
            : JSON.stringify(e.response.data.error);
      } else if (e?.message) {
        errorMsg = e.message;
      }
      console.error(
        "Comparison error:",
        errorMsg,
        "Benchmark ID:",
        selectedBenchmark
      );
      alert(
        `Failed to compare: ${errorMsg}\n\nMake sure the benchmark exists and try again.`
      );
    } finally {
      setLoadingComparison(false);
    }
  }

  const backendUrl =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
  const keyFrames = useMemo(() => data?.key_frames ?? [], [data]);

  const handleFileSelect = useCallback((selectedFile: File) => {
    if (!selectedFile.type.startsWith("video/")) {
      setUploadError("Please select a video file (.mp4, .mov, .avi, .mkv)");
      return;
    }
    setUploadFile(selectedFile);
    setUploadError(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        handleFileSelect(droppedFile);
      }
    },
    [handleFileSelect]
  );

  const handleUpload = () => {
    if (!uploadFile) {
      setUploadError("Please select a video to upload.");
      return;
    }
    if (!quickCameraAngle) {
      setUploadError("Select the camera angle for this serve.");
      return;
    }
    if (!quickServeType) {
      setUploadError("Select the serve type before uploading.");
      return;
    }
    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    const form = new FormData();
    form.append("file", uploadFile);
    form.append("camera_angle", quickCameraAngle);
    form.append("serve_type", quickServeType);

    axios
      .post("/api/upload", form, {
        onUploadProgress: (evt) => {
          if (evt.total) {
            setUploadProgress(Math.round((evt.loaded / evt.total) * 100));
          }
        },
      })
      .then((res) => {
        const videoId = res.data?.video_id || res.data?.videoId;
        if (!videoId) throw new Error("Invalid response from server");
        router.push(`/results/${videoId}`);
      })
      .catch((err: any) => {
        let errorMsg = "Upload failed";
        if (err?.response?.data?.detail) {
          const detail = err.response.data.detail;
          if (Array.isArray(detail)) {
            errorMsg = detail
              .map((d: any) => d.msg || JSON.stringify(d))
              .join(", ");
          } else if (typeof detail === "string") {
            errorMsg = detail;
          } else {
            errorMsg = JSON.stringify(detail);
          }
        } else if (err?.response?.data?.error) {
          errorMsg =
            typeof err.response.data.error === "string"
              ? err.response.data.error
              : JSON.stringify(err.response.data.error);
        } else if (err?.message) {
          errorMsg = err.message;
        }
        setUploadError(errorMsg);
        setUploading(false);
      });
  };

  return (
    <div className="space-y-6 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
          Results
        </h2>
        {!loading && (
          <button
            onClick={() => router.push("/upload")}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
          >
            📤 Upload New Video
          </button>
        )}
      </div>

      {!loading && (
        <div className="rounded-xl bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 backdrop-blur border border-blue-200 dark:border-blue-700/50 p-4">
          <h3 className="text-lg font-medium mb-3 text-slate-900 dark:text-slate-100">
            🎬 Analyze Another Serve
          </h3>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-4 text-center transition-all ${
              isDragOver
                ? "border-blue-500 bg-blue-100/50 dark:bg-blue-900/30 scale-[1.01]"
                : "border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500"
            }`}
          >
            {uploadFile ? (
              <div className="space-y-3">
                <p className="text-sm text-emerald-600 dark:text-emerald-400">
                  📹 {uploadFile.name} (
                  {(uploadFile.size / 1024 / 1024).toFixed(2)} MB)
                </p>
                <div className="flex gap-2 justify-center">
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    {uploading ? "Uploading…" : "Upload & Analyze"}
                  </button>
                  <button
                    onClick={() => {
                      setUploadFile(null);
                      setUploadError(null);
                    }}
                    disabled={uploading}
                    className="px-4 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-900 dark:text-slate-100 rounded-lg text-sm font-medium transition-colors"
                  >
                    Clear
                  </button>
                </div>
                {uploading && (
                  <div className="mt-2">
                    <div className="h-2 w-full bg-slate-200 dark:bg-slate-700 rounded">
                      <div
                        className="h-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded transition-all"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-xs mt-1 opacity-80 text-slate-700 dark:text-slate-300">
                      {uploadProgress}%
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                <svg
                  className={`mx-auto h-8 w-8 ${
                    isDragOver
                      ? "text-blue-500 dark:text-blue-400"
                      : "text-slate-400 dark:text-slate-500"
                  }`}
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p
                  className={`text-sm ${
                    isDragOver
                      ? "text-blue-600 dark:text-blue-300 font-medium"
                      : "text-slate-600 dark:text-slate-400"
                  }`}
                >
                  {isDragOver
                    ? "Drop your video here"
                    : "Drag and drop a video here or click to browse"}
                </p>
                <input
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                  onChange={(e) => {
                    const selectedFile = e.target.files?.[0];
                    if (selectedFile) handleFileSelect(selectedFile);
                  }}
                  className="hidden"
                  id="quick-upload"
                />
                <label
                  htmlFor="quick-upload"
                  className="inline-block mt-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white rounded-lg text-sm font-medium cursor-pointer transition-colors"
                >
                  Select Video File
                </label>
              </div>
            )}
            {uploadError && (
              <p className="text-red-600 dark:text-red-400 mt-2 text-sm">
                {String(uploadError)}
              </p>
            )}
          </div>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                Camera Angle
              </label>
              <select
                value={quickCameraAngle}
                onChange={(e) => setQuickCameraAngle(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select angle...</option>
                <option value="side">Side View</option>
                <option value="back">Back View</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1">
                Serve Type
              </label>
              <select
                value={quickServeType}
                onChange={(e) => setQuickServeType(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-sm text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select serve type...</option>
                <option value="jump_float_serve">Jump Float Serve</option>
                <option value="standing_float_serve">
                  Standing Float Serve
                </option>
                <option value="jump_topspin_serve">Jump Topspin Serve</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 space-y-6">
          <div className="relative w-32 h-32">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-28 h-28 rounded-full border-6 border-blue-500 border-t-transparent dark:border-blue-400 dark:border-t-transparent animate-spin"></div>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                <span className="text-white text-4xl">🏐</span>
              </div>
            </div>
          </div>
          <div className="text-center space-y-3">
            <h3 className="text-3xl font-semibold text-slate-900 dark:text-slate-100 animate-pulse">
              Analyzing Your Serve
            </h3>
            <div className="flex items-center justify-center gap-3 text-slate-600 dark:text-slate-400">
              <span className="flex gap-2">
                <span
                  className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: "0ms" }}
                ></span>
                <span
                  className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: "150ms" }}
                ></span>
                <span
                  className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: "300ms" }}
                ></span>
              </span>
              <span className="text-base">Processing video frames</span>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-500 mt-3">
              This may take a moment...
            </p>
          </div>
        </div>
      )}
      {error && (
        <p className="text-red-600 dark:text-red-400">{String(error)}</p>
      )}
      {data && !loading && (
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white/70 dark:bg-slate-800/70 backdrop-blur px-4 py-3 flex items-start gap-3 text-sm text-slate-700 dark:text-slate-200 shadow-sm">
            <span className="text-lg">💡</span>
            <p className="leading-snug">
              Every serve is different and unique to each player. Use these
              insights as guidance while you build your own style.
            </p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {keyFrames.map((kf) => {
              const imageUrl = kf.image_path?.startsWith("http")
                ? kf.image_path
                : kf.image_path?.startsWith("/uploads/")
                ? `${backendUrl}${kf.image_path}`
                : `${backendUrl}/uploads/${kf.image_path || ""}`;
              return (
                <div
                  key={`kf-${kf.frame_index}`}
                  className="rounded-xl overflow-hidden bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-500 transition-all cursor-pointer"
                  onClick={() => setSelectedImage(imageUrl)}
                >
                  <div className="relative bg-slate-100 dark:bg-slate-900 min-h-[200px] flex items-center justify-center">
                    <img
                      src={imageUrl}
                      alt={`frame ${kf.frame_index}`}
                      className="w-full h-auto object-contain"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                        const parent = (e.target as HTMLImageElement)
                          .parentElement;
                        if (parent && !parent.querySelector(".error-msg")) {
                          const errorMsg = document.createElement("div");
                          errorMsg.className =
                            "error-msg text-xs text-slate-500 dark:text-slate-400 p-4";
                          errorMsg.textContent = "Image not found";
                          parent.appendChild(errorMsg);
                        }
                      }}
                    />
                    {kf.label && (
                      <span
                        className={`absolute top-2 left-2 text-xs px-2 py-1 rounded-full text-white dark:text-white capitalize ${
                          labelColor[kf.label] || "bg-slate-600"
                        }`}
                      >
                        {kf.label.replace(/_/g, " ")}
                      </span>
                    )}
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/40">
                      <span className="text-slate-900 dark:text-white text-sm font-medium">
                        Click to enlarge
                      </span>
                    </div>
                  </div>
                  <div className="p-2 text-xs opacity-80">
                    Frame {kf.frame_index}
                  </div>
                </div>
              );
            })}
            {!keyFrames.length &&
              data.annotations?.map((a) => (
                <div
                  key={`ann-${a.frame_index}`}
                  className="rounded-xl overflow-hidden bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700"
                >
                  {a.preview_image ? (
                    <img
                      src={`${backendUrl}/uploads/${a.preview_image}`}
                      alt={`frame ${a.frame_index}`}
                    />
                  ) : (
                    <div className="p-4 text-xs">No image</div>
                  )}
                  <div className="p-2 text-xs opacity-80">
                    Frame {a.frame_index}
                  </div>
                </div>
              ))}
          </div>

          <ScrollReveal>
            <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4">
              <h3 className="font-medium mb-2 text-slate-900 dark:text-slate-100">
                Summary
              </h3>
              <div className="text-sm opacity-80 grid grid-cols-2 gap-2">
                <p>
                  Duration:{" "}
                  {data.duration_sec ? `${data.duration_sec.toFixed(2)}s` : "—"}
                </p>
                <p>
                  Total Frames: {data.frame_count ?? data.total_frames ?? "—"}
                </p>
                <p>
                  Frames Analyzed:{" "}
                  {data.frames_analyzed ?? data.frame_count ?? "—"}
                </p>
                <p>FPS: {data.fps ? `${data.fps.toFixed(1)}` : "—"}</p>
                {data.serve_type && (
                  <p className="mt-2 pt-2 border-t border-slate-300 dark:border-slate-600 col-span-2">
                    <span className="font-medium">Serve Type:</span>{" "}
                    {data.serve_type
                      .split("_")
                      .map(
                        (word: string) =>
                          word.charAt(0).toUpperCase() + word.slice(1)
                      )
                      .join(" ")}
                  </p>
                )}
                {data.camera_angle && (
                  <p className="col-span-2">
                    <span className="font-medium">Camera Angle:</span>{" "}
                    {data.camera_angle
                      .split("_")
                      .map(
                        (word: string) =>
                          word.charAt(0).toUpperCase() + word.slice(1)
                      )
                      .join(" ")}
                  </p>
                )}
              </div>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50/80 dark:bg-amber-950/30 backdrop-blur px-4 py-3 flex items-start gap-3 text-sm text-amber-800 dark:text-amber-200">
              <span className="text-lg shrink-0">ℹ️</span>
              <div className="leading-snug space-y-1">
                <p>
                  Data may not always show up due to lighting, occlusion, or
                  pose-detection limits. For best results, try recording in higher
                  quality or from a clearer angle (e.g. side view with your full body
                  and the ball visible).
                </p>
                <p>
                  <strong>Accuracy tip:</strong> Add your height when uploading—it
                  improves scale calibration so contact and elbow heights are more
                  reliable.
                </p>
              </div>
            </div>
          </ScrollReveal>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div>
              <ScrollReveal>
                {data?.feedback ? (
                  <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4">
                    <h3 className="font-medium mb-3 text-slate-900 dark:text-slate-100">
                      📈 Detailed Metrics
                    </h3>
                    {(!userMetrics || Object.keys(userMetrics).length === 0) && (
                      <p className="text-sm text-amber-700 dark:text-amber-400 mb-3 rounded-md bg-amber-50 dark:bg-amber-950/40 px-3 py-2 border border-amber-200 dark:border-amber-800">
                        We couldn&apos;t extract metrics from this video. Use a
                        clear side view with your full body and the ball visible,
                        add your height on upload for better scale, then
                        re-analyze.
                      </p>
                    )}
                    <div className="space-y-3 text-sm">
                      {/* 1. Toss → Contact Time */}
                      <div className="border border-slate-200 dark:border-slate-600 rounded-lg p-3 bg-slate-100/50 dark:bg-slate-900/30">
                        <div className="font-medium text-slate-900 dark:text-slate-100 mb-2">
                          Toss → Contact Time
                        </div>
                        <div className="text-xs">
                          <div className="font-mono text-slate-900 dark:text-white">
                            {userMetrics?.toss_contact_time_s !== undefined
                              ? formatSeconds(
                                  userMetrics.toss_contact_time_s,
                                  2
                                )
                              : "—"}
                          </div>
                        </div>
                      </div>
                      {/* 2. Contact (head to hand) */}
                      <div className="border border-slate-200 dark:border-slate-600 rounded-lg p-3 bg-slate-100/50 dark:bg-slate-900/30">
                        <div className="font-medium text-slate-900 dark:text-slate-100 mb-2">
                          Contact (head to hand)
                        </div>
                        <div className="text-xs">
                          <div className="font-mono text-slate-900 dark:text-white">
                            {userMetrics?.contact_height_m !== undefined
                              ? formatInches(userMetrics.contact_height_m)
                              : "—"}
                          </div>
                        </div>
                      </div>
                      {/* 3. Elbow (head to elbow) */}
                      <div className="border border-slate-200 dark:border-slate-600 rounded-lg p-3 bg-slate-100/50 dark:bg-slate-900/30">
                        <div className="font-medium text-slate-900 dark:text-slate-100 mb-2">
                          Elbow (head to elbow)
                        </div>
                        <div className="text-xs">
                          <div className="font-mono text-slate-900 dark:text-white">
                            {userMetrics?.elbow_height_m !== undefined
                              ? formatInches(userMetrics.elbow_height_m)
                              : "—"}
                          </div>
                        </div>
                      </div>
                      {/* 4. Toss Height */}
                      <div className="border border-slate-200 dark:border-slate-600 rounded-lg p-3 bg-slate-100/50 dark:bg-slate-900/30">
                        <div className="font-medium text-slate-900 dark:text-slate-100 mb-2">
                          Toss Height
                        </div>
                        <div className="text-xs">
                          <div className="font-mono text-slate-900 dark:text-white">
                            {userMetrics?.toss_height_m !== undefined
                              ? formatInches(userMetrics.toss_height_m)
                              : "—"}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4 flex flex-col justify-center">
                    <h3 className="font-medium mb-3 text-slate-900 dark:text-slate-100">
                      📈 Detailed Metrics
                    </h3>
                    <p className="text-sm opacity-60">
                      Detailed metrics will appear here once the analysis
                      extracts enough pose data from your serve.
                    </p>
                  </div>
                )}
              </ScrollReveal>
            </div>

            <div>
              <ScrollReveal delay={150}>
                {/* Automatic Professional Comparison - only show match when we have pro benchmarks */}
                {data.pro_comparison?.best_match &&
                (data.pro_comparison?.summary?.total_compared ?? 0) > 0 ? (
                  <div className="rounded-xl bg-gradient-to-br from-purple-900/40 to-blue-900/40 backdrop-blur border border-purple-700/50 p-4 h-full">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-semibold text-lg flex items-center gap-2 text-slate-900 dark:text-slate-100">
                        ⭐ Professional Comparison
                      </h3>
                      <div className="px-3 py-1 bg-purple-700/50 rounded-full text-sm">
                        {data.pro_comparison.summary?.best_match_score?.toFixed(
                          0
                        ) || 0}
                        % Match
                      </div>
                    </div>
                    <p className="text-sm opacity-80 mb-4">
                      Compared against{" "}
                      {data.pro_comparison.summary?.total_compared || 0}{" "}
                      professional{" "}
                      {data.serve_type
                        ? data.serve_type
                            .split("_")
                            .map(
                              (word: string) =>
                                word.charAt(0).toUpperCase() + word.slice(1)
                            )
                            .join(" ")
                        : "serve"}{" "}
                      benchmarks
                      {data.camera_angle
                        ? ` from ${data.camera_angle
                            .split("_")
                            .map(
                              (word: string) =>
                                word.charAt(0).toUpperCase() + word.slice(1)
                            )
                            .join(" ")} view`
                        : ""}
                      .
                    </p>
                    <div className="rounded-lg bg-slate-800/40 border border-slate-600/50 px-3 py-2 mb-4 text-xs text-slate-400 dark:text-slate-500">
                      <strong className="text-slate-300 dark:text-slate-400">About these comparisons:</strong> Pro
                      reference values come from literature and may use different measurement methods or definitions. Our
                      scale depends on camera angle and player height—add your height on upload for more accurate results.
                      Use these as guidance, not absolute targets.
                    </div>
                    {data.pro_comparison.best_match.benchmark.description && (
                      <p className="text-xs opacity-60 mb-4 italic">
                        {data.pro_comparison.best_match.benchmark.description}
                      </p>
                    )}
                    {(data.pro_comparison.summary?.best_match_score ?? 0) ===
                      0 &&
                      (!data.pro_comparison.best_match.comparison
                        ?.differences ||
                        Object.keys(
                          data.pro_comparison.best_match.comparison.differences
                        ).length === 0) && (
                        <p className="text-xs text-amber-600 dark:text-amber-400 mb-4">
                          No metrics from your analysis could be compared yet.
                          Re-analyze with a clear side view and full body
                          visible so we can compute heights and compare to this
                          benchmark.
                        </p>
                      )}
                    <div className="space-y-3 text-sm">
                      {data.pro_comparison.best_match.comparison &&
                        data.pro_comparison.best_match.comparison.differences &&
                        (
                          [
                            "toss_contact_time_s",
                            "contact_height_m",
                            "elbow_height_m",
                            "toss_height_m",
                          ] as const
                        )
                          .filter(
                            (key) =>
                              data.pro_comparison?.best_match?.comparison
                                ?.differences[key] != null
                          )
                          .map((key) => {
                            const diff =
                              data.pro_comparison.best_match.comparison
                                .differences[key] as
                              | { percent?: number; benchmark_missing?: boolean }
                              | undefined;
                            const isBenchmarkMissing =
                              diff && "benchmark_missing" in diff && diff.benchmark_missing;
                            const userVal =
                              data.pro_comparison.best_match.comparison.user[
                                key
                              ];
                            const benchVal =
                              data.pro_comparison.best_match.comparison
                                .benchmark[key];
                            const rating =
                              data.pro_comparison.best_match.comparison.ratings[
                                key
                              ];
                            const labels: Record<string, string> = {
                              toss_contact_time_s: "Toss → Contact Time",
                              contact_height_m: "Contact (head to hand)",
                              elbow_height_m: "Elbow (head to elbow)",
                              toss_height_m: "Toss Height",
                            };
                            const formatVal = (v: unknown) => {
                              if (typeof v !== "number")
                                return String(v ?? "—");
                              if (key === "toss_contact_time_s")
                                return formatSeconds(v, 2);
                              if (key.endsWith("_m")) return formatInches(v);
                              return v.toFixed(2);
                            };
                            return (
                              <div
                                key={key}
                                className="border border-purple-700/50 rounded-lg p-3 bg-slate-100/50 dark:bg-slate-900/30"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <div className="font-medium text-slate-900 dark:text-slate-100">
                                    {labels[key] || key}
                                  </div>
                                  <div
                                    className={`text-xs px-2 py-1 rounded ${
                                      rating === "no_pro_data"
                                        ? "bg-slate-700/50 text-slate-300 dark:text-slate-400"
                                        : rating === "excellent"
                                        ? "bg-emerald-900/50 text-emerald-300"
                                        : rating === "good"
                                        ? "bg-blue-900/50 text-blue-300"
                                        : rating === "needs_improvement"
                                        ? "bg-amber-900/50 text-amber-300"
                                        : "bg-red-900/50 text-red-300"
                                    }`}
                                  >
                                    {rating === "no_pro_data"
                                      ? "Pro data not available"
                                      : rating === "excellent"
                                      ? "✅ Excellent"
                                      : rating === "good"
                                      ? "✅ Good"
                                      : rating === "needs_improvement"
                                      ? "⚠️ Needs Improvement"
                                      : "❌ Far Off"}
                                  </div>
                                </div>
                                <div className="grid grid-cols-3 gap-2 text-xs">
                                  <div>
                                    <div className="opacity-60">You:</div>
                                    <div className="font-mono text-slate-900 dark:text-white">
                                      {formatVal(userVal)}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="opacity-60">Pro:</div>
                                    <div className="font-mono text-purple-700 dark:text-purple-300">
                                      {isBenchmarkMissing ? "—" : formatVal(benchVal)}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="opacity-60">
                                      Difference:
                                    </div>
                                    <div
                                      className={`font-mono ${
                                        isBenchmarkMissing
                                          ? "text-slate-500 dark:text-slate-400"
                                          : (diff?.percent ?? 0) >= 0
                                          ? "text-emerald-600 dark:text-emerald-400"
                                          : "text-red-600 dark:text-red-400"
                                      }`}
                                    >
                                      {isBenchmarkMissing
                                        ? "—"
                                        : diff?.percent != null
                                        ? `${(diff.percent as number) > 0 ? "+" : ""}${(diff.percent as number).toFixed(1)}%`
                                        : "—"}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4 h-full flex flex-col justify-center">
                    <h3 className="font-semibold text-lg text-slate-900 dark:text-slate-100 mb-2">
                      ⭐ Professional Comparison
                    </h3>
                    <p className="text-sm opacity-60">
                      No professional benchmarks to compare against. Your
                      metrics above are from your serve only. Add pro benchmarks
                      (e.g. from screenshots) to see a comparison and match %.
                    </p>
                  </div>
                )}
              </ScrollReveal>
            </div>
          </div>

          {/* Benchmark Actions */}
          <ScrollReveal delay={200}>
            <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4">
              <h3 className="font-medium mb-3 text-slate-900 dark:text-slate-100">
                🎯 Custom Benchmark & Comparison
              </h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => setShowBenchmarkModal(true)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
                >
                  💾 Save as Benchmark
                </button>
                {benchmarks.length > 0 && (
                  <div className="flex gap-2 flex-wrap items-center">
                    <select
                      value={selectedBenchmark || ""}
                      onChange={(e) => setSelectedBenchmark(e.target.value)}
                      className="px-3 py-2 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-sm text-slate-900 dark:text-slate-100"
                    >
                      <option value="">Select benchmark...</option>
                      {benchmarks.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.name}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={compareWithBenchmark}
                      disabled={!selectedBenchmark || loadingComparison}
                      className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
                    >
                      {loadingComparison ? "Comparing..." : "Compare"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </ScrollReveal>

          {/* Comparison Results */}
          {comparisonData && (
            <ScrollReveal delay={250}>
              <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-slate-900 dark:text-slate-100">
                    📊 Comparison: Your Serve vs {comparisonData.benchmark.name}
                  </h3>
                  {comparisonData.comparison?.overall_score !== undefined && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs opacity-60">Match Score:</span>
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-semibold ${
                          comparisonData.comparison.overall_score >= 80
                            ? "bg-emerald-500/20 text-emerald-700 dark:text-emerald-400"
                            : comparisonData.comparison.overall_score >= 60
                            ? "bg-blue-500/20 text-blue-700 dark:text-blue-400"
                            : comparisonData.comparison.overall_score >= 40
                            ? "bg-amber-500/20 text-amber-700 dark:text-amber-400"
                            : "bg-red-500/20 text-red-700 dark:text-red-400"
                        }`}
                      >
                        {comparisonData.comparison.overall_score.toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
                <div className="space-y-3 text-sm">
                  {comparisonData.comparison &&
                    comparisonData.comparison.differences &&
                    (
                      [
                        "toss_contact_time_s",
                        "contact_height_m",
                        "elbow_height_m",
                        "toss_height_m",
                      ] as const
                    )
                      .filter(
                        (key) =>
                          comparisonData.comparison.differences[key] != null
                      )
                      .map((key) => {
                        const diff = comparisonData.comparison.differences[key];
                        const userVal = comparisonData.comparison.user[key];
                        const benchVal =
                          comparisonData.comparison.benchmark[key];
                        const rating = comparisonData.comparison.ratings[key];
                        const labels: Record<string, string> = {
                          toss_contact_time_s: "Toss → Contact Time",
                          contact_height_m: "Contact (head to hand)",
                          elbow_height_m: "Elbow (head to elbow)",
                          toss_height_m: "Toss Height",
                        };
                        const formatVal = (v: unknown) => {
                          if (typeof v !== "number") return String(v ?? "—");
                          if (key === "toss_contact_time_s")
                            return formatSeconds(v, 2);
                          if (key.endsWith("_m")) return formatInches(v);
                          return v.toFixed(2);
                        };
                        return (
                          <div
                            key={key}
                            className="border border-slate-200 dark:border-slate-700 rounded-lg p-3"
                          >
                            <div className="font-medium mb-2">
                              {labels[key] || key}
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              <div>
                                <div className="opacity-60">You:</div>
                                <div className="font-mono">
                                  {formatVal(userVal)}
                                </div>
                              </div>
                              <div>
                                <div className="opacity-60">Benchmark:</div>
                                <div className="font-mono">
                                  {formatVal(benchVal)}
                                </div>
                              </div>
                              <div>
                                <div className="opacity-60">Difference:</div>
                                <div
                                  className={`font-mono ${
                                    diff.percent >= 0
                                      ? "text-emerald-600 dark:text-emerald-400"
                                      : "text-red-600 dark:text-red-400"
                                  }`}
                                >
                                  {diff.percent > 0 ? "+" : ""}
                                  {diff.percent.toFixed(1)}%
                                </div>
                                <div
                                  className={`text-xs mt-1 ${
                                    rating === "excellent"
                                      ? "text-emerald-600 dark:text-emerald-400"
                                      : rating === "good"
                                      ? "text-blue-600 dark:text-blue-400"
                                      : rating === "needs_improvement"
                                      ? "text-amber-600 dark:text-amber-400"
                                      : "text-red-600 dark:text-red-400"
                                  }`}
                                >
                                  {rating === "excellent"
                                    ? "✅ Excellent match"
                                    : rating === "good"
                                    ? "✅ Good"
                                    : rating === "needs_improvement"
                                    ? "⚠️ Needs improvement"
                                    : "❌ Far off"}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                </div>
              </div>
            </ScrollReveal>
          )}

          <ScrollReveal delay={300}>
            <div className="rounded-xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-4">
              <h3 className="font-medium mb-3 text-slate-900 dark:text-slate-100">
                Feedback
              </h3>

              {(() => {
                // Hide tips about removed metrics (elbow extension, knee bend, arm range, stance)
                const isRemovedMetricTip = (t: string) =>
                  /elbow extension|knee bend|arm range|stance width|narrow stance|Great arm|Moderate arm|Limited arm|Stiff legs|excellent elbow|good elbow|Limited elbow/i.test(
                    t
                  );
                const tips = (data.feedback?.tips ?? []).filter(
                  (t: string) => !isRemovedMetricTip(t)
                );
                const recs = (data.feedback?.recommendations ?? []).filter(
                  (r: string) => !isRemovedMetricTip(r)
                );
                return (
                  <>
                    {tips.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium mb-2 opacity-80">
                          What's Working:
                        </h4>
                        <ul className="list-disc pl-5 space-y-1 text-sm">
                          {tips
                            .filter((t: string) => t.includes("✅"))
                            .map((t: string, i: number) => (
                              <li
                                key={i}
                                className="text-emerald-600 dark:text-emerald-400"
                              >
                                {t}
                              </li>
                            ))}
                        </ul>
                      </div>
                    )}

                    {tips.length > 0 &&
                      tips.some((t: string) => !t.includes("✅")) && (
                        <div className="mb-4">
                          <h4 className="text-sm font-medium mb-2 opacity-80">
                            Areas to Improve:
                          </h4>
                          <ul className="list-disc pl-5 space-y-1 text-sm">
                            {tips
                              .filter((t: string) => !t.includes("✅"))
                              .map((t: string, i: number) => (
                                <li
                                  key={i}
                                  className={
                                    t.includes("⚠️")
                                      ? "text-amber-600 dark:text-amber-400"
                                      : t.includes("❌")
                                      ? "text-red-600 dark:text-red-400"
                                      : "text-slate-700 dark:text-slate-300"
                                  }
                                >
                                  {t}
                                </li>
                              ))}
                          </ul>
                        </div>
                      )}

                    {recs.length > 0 && (
                      <div className="border-t border-slate-200 dark:border-slate-700 pt-4 mt-4">
                        <h4 className="text-sm font-medium mb-3 text-blue-600 dark:text-blue-400">
                          💡 Recommendations & Next Steps:
                        </h4>
                        <ul className="space-y-2 text-sm">
                          {recs.map((rec: string, i: number) => (
                            <li
                              key={i}
                              className="text-blue-700 dark:text-blue-300 flex items-start gap-2"
                            >
                              <span className="text-blue-600 dark:text-blue-500 mt-0.5">
                                •
                              </span>
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {tips.length === 0 && recs.length === 0 && (
                      <p className="text-sm opacity-60">
                        No feedback available
                      </p>
                    )}
                  </>
                );
              })()}
            </div>
          </ScrollReveal>
        </div>
      )}

      {/* Image Modal/Lightbox */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-5xl max-h-[90vh] w-full h-full flex items-center justify-center">
            <img
              src={selectedImage}
              alt="Enlarged frame"
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-4 right-4 text-white bg-slate-800/80 dark:bg-slate-800/80 hover:bg-slate-700 dark:hover:bg-slate-700 rounded-full p-2 transition-colors"
              aria-label="Close"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Benchmark Save Modal */}
      {showBenchmarkModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-4"
          onClick={() => setShowBenchmarkModal(false)}
        >
          <div
            className="bg-white dark:bg-slate-800 rounded-xl p-6 max-w-md w-full border border-slate-200 dark:border-slate-700"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">
              Save as Benchmark
            </h3>
            <p className="text-sm opacity-80 mb-4 text-slate-700 dark:text-slate-300">
              Save this professional serve as a benchmark to compare against
              future serves.
            </p>
            <input
              type="text"
              value={benchmarkName}
              onChange={(e) => setBenchmarkName(e.target.value)}
              placeholder="e.g., 'Pro Player Serve' or 'Olympic Gold Medalist'"
              className="w-full px-4 py-2 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-900 dark:text-slate-100"
              onKeyDown={(e) => {
                if (e.key === "Enter" && benchmarkName.trim()) {
                  saveAsBenchmark();
                }
              }}
            />
            <div className="flex gap-3">
              <button
                onClick={saveAsBenchmark}
                disabled={!benchmarkName.trim() || savingBenchmark}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:opacity-50 rounded-lg font-medium transition-colors"
              >
                {savingBenchmark ? "Saving..." : "Save"}
              </button>
              <button
                onClick={() => {
                  setShowBenchmarkModal(false);
                  setBenchmarkName("");
                }}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
