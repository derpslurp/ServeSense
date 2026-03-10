"use client";

import { useState } from "react";
import axios from "axios";
import Link from "next/link";

type Metrics = {
  toss_contact_time_s?: number;
  contact_height_m?: number;
  elbow_height_m?: number;
  toss_height_m?: number;
};

export default function CreateBenchmarkPage() {
  const [imageToss, setImageToss] = useState<File | null>(null);
  const [imageContact, setImageContact] = useState<File | null>(null);
  const [playerHeightCm, setPlayerHeightCm] = useState("");
  const [tossContactTimeS, setTossContactTimeS] = useState("");
  const [serveType, setServeType] = useState("");
  const [cameraAngle, setCameraAngle] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  async function handleAnalyze(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setMetrics(null);
    if (!imageToss || !imageContact) {
      setError("Please select both a toss frame and a contact frame.");
      return;
    }
    setAnalyzing(true);
    try {
      const form = new FormData();
      form.append("image_toss", imageToss, imageToss.name);
      form.append("image_contact", imageContact, imageContact.name);
      if (playerHeightCm.trim())
        form.append("player_height_cm", playerHeightCm.trim());
      if (tossContactTimeS.trim())
        form.append("toss_contact_time_s", tossContactTimeS.trim());
      const res = await axios.post("/api/analyze-frames", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.data?.error) {
        setError(res.data.error);
        return;
      }
      setMetrics(res.data?.metrics ?? null);
    } catch (err: unknown) {
      const axErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(
        axErr?.response?.data?.detail || axErr?.message || "Analysis failed"
      );
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleSaveBenchmark(e: React.FormEvent) {
    e.preventDefault();
    if (!benchmarkName.trim() || !serveType || !cameraAngle) {
      setError("Name, serve type, and camera angle are required to save.");
      return;
    }
    if (!imageToss || !imageContact) {
      setError("Both toss and contact images are required to save with reference images.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("image_toss", imageToss, imageToss.name);
      form.append("image_contact", imageContact, imageContact.name);
      form.append("name", benchmarkName.trim());
      form.append("serve_type", serveType);
      form.append("camera_angle", cameraAngle);
      if (playerHeightCm.trim()) form.append("player_height_cm", playerHeightCm.trim());
      if (tossContactTimeS.trim()) form.append("toss_contact_time_s", tossContactTimeS.trim());
      await axios.post("/api/pro-benchmarks/from-frames", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSaved(true);
    } catch (err: unknown) {
      const axErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(
        axErr?.response?.data?.detail ||
          axErr?.message ||
          "Failed to save benchmark"
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="max-w-2xl mx-auto px-4 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Link
            href="/"
            className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
          >
            ← Home
          </Link>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
            Create benchmark from frames
          </h1>
        </div>

        <p className="text-slate-600 dark:text-slate-400 mb-6">
          Upload two screenshots from a video: one at the <strong>toss</strong>{" "}
          (ball at peak or release) and one at <strong>contact</strong> (hit).
          We'll extract metrics and let you save them as a pro benchmark.
        </p>

        <form onSubmit={handleAnalyze} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Toss frame (image)
            </label>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
              onChange={(e) => setImageToss(e.target.files?.[0] ?? null)}
              className="w-full text-sm text-slate-700 dark:text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-100 dark:file:bg-blue-900/40 file:text-blue-700 dark:file:text-blue-300"
            />
            {imageToss && (
              <p className="text-xs mt-1 opacity-80">{imageToss.name}</p>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Contact frame (image)
            </label>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
              onChange={(e) => setImageContact(e.target.files?.[0] ?? null)}
              className="w-full text-sm text-slate-700 dark:text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-100 dark:file:bg-blue-900/40 file:text-blue-700 dark:file:text-blue-300"
            />
            {imageContact && (
              <p className="text-xs mt-1 opacity-80">{imageContact.name}</p>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Player height (optional, cm)
            </label>
            <input
              type="number"
              min={100}
              max={250}
              placeholder="e.g. 175"
              value={playerHeightCm}
              onChange={(e) => setPlayerHeightCm(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Improves height accuracy (toss, contact, elbow).
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Toss → contact time (optional, seconds)
            </label>
            <input
              type="number"
              min={0}
              step={0.1}
              placeholder="e.g. 0.9"
              value={tossContactTimeS}
              onChange={(e) => setTossContactTimeS(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              If you know the time between toss and contact (e.g. from video),
              enter it here.
            </p>
          </div>
          <button
            type="submit"
            disabled={analyzing || !imageToss || !imageContact}
            className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg font-medium"
          >
            {analyzing ? "Analyzing…" : "Analyze frames"}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        {metrics && Object.keys(metrics).length > 0 && (
          <div className="mt-8 p-4 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-3">
              Extracted metrics
            </h2>
            <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
              {metrics.toss_height_m != null && (
                <li>
                  <strong>Toss height:</strong>{" "}
                  {(metrics.toss_height_m * 39.3701).toFixed(1)} in
                </li>
              )}
              {metrics.contact_height_m != null && (
                <li>
                  <strong>Contact (head to hand):</strong>{" "}
                  {(metrics.contact_height_m * 39.3701).toFixed(1)} in
                </li>
              )}
              {metrics.elbow_height_m != null && (
                <li>
                  <strong>Elbow (head to elbow):</strong>{" "}
                  {(metrics.elbow_height_m * 39.3701).toFixed(1)} in
                </li>
              )}
              {metrics.toss_contact_time_s != null && (
                <li>
                  <strong>Toss → contact time:</strong>{" "}
                  {metrics.toss_contact_time_s.toFixed(2)} s
                </li>
              )}
            </ul>

            {!saved ? (
              <form onSubmit={handleSaveBenchmark} className="mt-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Save as pro benchmark
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Pro jump float side"
                    value={benchmarkName}
                    onChange={(e) => setBenchmarkName(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Serve type
                  </label>
                  <select
                    value={serveType}
                    onChange={(e) => setServeType(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  >
                    <option value="">Select…</option>
                    <option value="jump_float_serve">Jump Float Serve</option>
                    <option value="standing_float_serve">
                      Standing Float Serve
                    </option>
                    <option value="jump_topspin_serve">
                      Jump Topspin Serve
                    </option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Camera angle
                  </label>
                  <select
                    value={cameraAngle}
                    onChange={(e) => setCameraAngle(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  >
                    <option value="">Select…</option>
                    <option value="side">Side View</option>
                    <option value="back">Back View</option>
                  </select>
                </div>
                <button
                  type="submit"
                  disabled={
                    saving ||
                    !benchmarkName.trim() ||
                    !serveType ||
                    !cameraAngle
                  }
                  className="w-full px-4 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded-lg font-medium"
                >
                  {saving ? "Saving…" : "Save as pro benchmark"}
                </button>
              </form>
            ) : (
              <p className="mt-4 text-emerald-600 dark:text-emerald-400 font-medium">
                Benchmark saved. You can use it in comparisons on the results
                page.
              </p>
            )}
          </div>
        )}

        {metrics && Object.keys(metrics).length === 0 && !error && (
          <p className="mt-4 text-amber-600 dark:text-amber-400 text-sm">
            No metrics could be extracted. Ensure both frames show the player
            clearly and (for heights) the ball is visible in toss and contact
            frames.
          </p>
        )}
      </div>
    </div>
  );
}
