"use client";

import { useState, useCallback } from "react";
import axios from "axios";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ScrollReveal from "../../components/ScrollReveal";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [cameraAngle, setCameraAngle] = useState<string>("");
  const [serveType, setServeType] = useState<string>("");
  const [playerHeightCm, setPlayerHeightCm] = useState<string>("");

  const handleFile = useCallback((selectedFile: File) => {
    const validMimeTypes = [
      "video/mp4",
      "video/quicktime",
      "video/x-msvideo",
      "video/x-matroska",
      "video/avi",
    ];
    const validExtensions = [".mp4", ".mov", ".avi", ".mkv"];
    const fileName = selectedFile.name.toLowerCase();
    const hasValidExtension = validExtensions.some((ext) =>
      fileName.endsWith(ext)
    );
    const hasValidMimeType =
      selectedFile.type &&
      (validMimeTypes.includes(selectedFile.type) ||
        selectedFile.type.startsWith("video/"));

    if (!hasValidMimeType && !hasValidExtension) {
      setError("Please select a video file (.mp4, .mov, .avi, .mkv)");
      return;
    }
    setFile(selectedFile);
    setError(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) handleFile(droppedFile);
    },
    [handleFile]
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!file) {
      setError("Please choose a video file");
      return;
    }
    if (!cameraAngle) {
      setError("Please select the camera angle");
      return;
    }
    if (!serveType) {
      setError("Please select the serve type");
      return;
    }
    setLoading(true);
    setProgress(0);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("camera_angle", cameraAngle);
      form.append("serve_type", serveType);
      const cm = playerHeightCm.trim() ? parseFloat(playerHeightCm) : NaN;
      if (Number.isFinite(cm) && cm > 0 && cm < 300) {
        form.append("player_height_cm", String(cm));
      }
      const res = await axios.post("/api/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (evt) => {
          if (evt.total) {
            setProgress(Math.round((evt.loaded / evt.total) * 100));
          }
        },
      });
      const videoId = res.data?.video_id || res.data?.videoId;
      if (!videoId) throw new Error("Invalid response from server");
      router.push(`/results/${videoId}`);
    } catch (err: unknown) {
      const axErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(axErr?.response?.data?.detail || axErr?.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 via-fuchsia-600/10 to-emerald-600/20 rounded-3xl blur-2xl -z-10" />
      <div className="flex justify-end mb-2">
        <Link
          href="/create-benchmark"
          className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
        >
          Create benchmark from frames →
        </Link>
      </div>

      <ScrollReveal>
        <div className="mb-6 rounded-xl bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border border-amber-200 dark:border-amber-700/50 p-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <svg
                className="w-6 h-6 text-amber-600 dark:text-amber-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-amber-900 dark:text-amber-200 mb-1">
                💡 Pro Tip
              </h3>
              <p className="text-sm text-amber-800 dark:text-amber-300">
                Record from different angles for better results! Side view and
                front view help capture your form, technique, and body mechanics
                more accurately.
              </p>
            </div>
          </div>
        </div>
      </ScrollReveal>

      <ScrollReveal delay={100}>
        <div className="rounded-2xl bg-slate-50/60 dark:bg-slate-800/60 backdrop-blur border border-slate-200 dark:border-slate-700 p-6">
          <h2 className="text-xl font-medium mb-4 text-slate-900 dark:text-slate-100">
            Upload your serve video
          </h2>
          <form onSubmit={onSubmit} className="space-y-4">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                isDragging
                  ? "border-blue-500 bg-blue-500/10 dark:bg-blue-500/20 scale-[1.02]"
                  : "border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500"
              }`}
            >
              <div className="space-y-2">
                <svg
                  className={`mx-auto h-12 w-12 ${
                    isDragging
                      ? "text-blue-500 dark:text-blue-400"
                      : "text-slate-500 dark:text-slate-400"
                  }`}
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                  aria-hidden="true"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <div>
                  <p
                    className={`text-sm ${
                      isDragging
                        ? "text-blue-600 dark:text-blue-300 font-medium"
                        : "text-slate-700 dark:text-slate-300"
                    }`}
                  >
                    {isDragging
                      ? "Drop your video here"
                      : "Drag and drop your video here"}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    or click to browse
                  </p>
                </div>
                {file && (
                  <p className="text-sm text-emerald-400 mt-2">
                    📹 {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </p>
                )}
              </div>
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/avi,.mp4,.mov,.avi,.mkv"
                onChange={(e) => {
                  const selectedFile = e.target.files?.[0];
                  if (selectedFile) handleFile(selectedFile);
                }}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="mt-4 inline-block px-4 py-2 text-sm bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 rounded cursor-pointer transition-colors text-slate-900 dark:text-slate-100"
              >
                Select Video File
              </label>
            </div>

            <div className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-950/30 p-3">
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                <span className="inline-flex items-center gap-1.5">
                  Your height
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-700 dark:text-amber-400 font-medium">
                    Recommended for accuracy
                  </span>
                </span>
              </label>
              <input
                type="number"
                min={100}
                max={250}
                step={1}
                placeholder="e.g. 175 (cm)"
                value={playerHeightCm}
                onChange={(e) => setPlayerHeightCm(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                Enter your height in cm for accurate toss/contact/elbow measurements. Without it, we assume 1.75 m.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Camera Angle
              </label>
              <select
                value={cameraAngle}
                onChange={(e) => setCameraAngle(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select camera angle...</option>
                <option value="side">Side View</option>
                <option value="back">Back View</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Serve Type
              </label>
              <select
                value={serveType}
                onChange={(e) => setServeType(e.target.value)}
                className="w-full px-4 py-2.5 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select serve type...</option>
                <option value="jump_float_serve">Jump Float Serve</option>
                <option value="standing_float_serve">
                  Standing Float Serve
                </option>
                <option value="jump_topspin_serve">Jump Topspin Serve</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading || !file || !cameraAngle || !serveType}
              className="w-full px-5 py-2.5 bg-gradient-to-r from-blue-600 to-fuchsia-600 rounded text-white disabled:opacity-50 disabled:cursor-not-allowed shadow hover:shadow-lg transition-all"
            >
              {loading ? "Uploading…" : "Upload & Analyze"}
            </button>
          </form>
          {loading && (
            <div className="mt-3">
              <div className="h-2 w-full bg-slate-200 dark:bg-slate-700 rounded">
                <div
                  className="h-2 bg-gradient-to-r from-blue-500 to-fuchsia-500 rounded transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-xs mt-1 opacity-80 text-slate-700 dark:text-slate-300">
                {progress}%
              </p>
            </div>
          )}
          {error && (
            <p className="text-red-600 dark:text-red-400 mt-3">{error}</p>
          )}
        </div>
      </ScrollReveal>

      <ScrollReveal delay={200}>
        <div className="mt-8">
          <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">
            ✨ Features
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border border-blue-200 dark:border-blue-700/50 p-5 transition-all duration-300 hover:from-blue-100 hover:to-blue-200 dark:hover:from-blue-800/30 dark:hover:to-blue-700/30 hover:border-blue-400 dark:hover:border-blue-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-blue-500 dark:bg-blue-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Video Analysis
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    AI-powered pose detection and technique analysis using
                    advanced computer vision to track your movement throughout
                    the serve
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border border-purple-200 dark:border-purple-700/50 p-5 transition-all duration-300 hover:from-purple-100 hover:to-purple-200 dark:hover:from-purple-800/30 dark:hover:to-purple-700/30 hover:border-purple-400 dark:hover:border-purple-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-purple-500 dark:bg-purple-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Detailed Metrics
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Track elbow extension, knee bend, arm range, stance width,
                    and more with precise measurements
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-br from-emerald-50 to-emerald-100 dark:from-emerald-900/20 dark:to-emerald-800/20 border border-emerald-200 dark:border-emerald-700/50 p-5 transition-all duration-300 hover:from-emerald-100 hover:to-emerald-200 dark:hover:from-emerald-800/30 dark:hover:to-emerald-700/30 hover:border-emerald-400 dark:hover:border-emerald-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-emerald-500 dark:bg-emerald-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    AI Feedback
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Get personalized tips and actionable recommendations to
                    improve your serve technique
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 border border-amber-200 dark:border-amber-700/50 p-5 transition-all duration-300 hover:from-amber-100 hover:to-amber-200 dark:hover:from-amber-800/30 dark:hover:to-amber-700/30 hover:border-amber-400 dark:hover:border-amber-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-amber-500 dark:bg-amber-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Pro Comparison
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Compare your technique against professional serves and see
                    how you measure up
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-br from-rose-50 to-rose-100 dark:from-rose-900/20 dark:to-rose-800/20 border border-rose-200 dark:border-rose-700/50 p-5 transition-all duration-300 hover:from-rose-100 hover:to-rose-200 dark:hover:from-rose-800/30 dark:hover:to-rose-700/30 hover:border-rose-400 dark:hover:border-rose-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-rose-500 dark:bg-rose-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Annotated Frames
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Visualize key moments with pose overlays showing joint
                    positions and angles
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-900/20 dark:to-teal-800/20 border border-teal-200 dark:border-teal-700/50 p-5 transition-all duration-300 hover:from-teal-100 hover:to-teal-200 dark:hover:from-teal-800/30 dark:hover:to-teal-700/30 hover:border-teal-400 dark:hover:border-teal-500 cursor-pointer">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-teal-500 dark:bg-teal-600 flex items-center justify-center">
                  <svg
                    className="w-7 h-7 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    Custom Benchmarks
                  </h4>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Save your best serves as benchmarks and track your
                    improvement over time
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </ScrollReveal>
    </div>
  );
}
