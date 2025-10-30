"use client"

import { useEffect, useState } from 'react'
import axios from 'axios'
import { useParams } from 'next/navigation'

interface AnalysisResponse {
	video_id?: string
	duration_sec?: number
	frame_count?: number
	annotations?: { frame_index: number; preview_image?: string; image_path?: string }[]
	key_moments?: Record<string, number>
	key_frames?: { frame_index: number; image_path: string }[]
	feedback?: { tips: string[]; output_dir?: string; pose_sample?: any[] }
}

export default function ResultsPage() {
	const params = useParams<{ id: string }>()
	const id = params?.id
	const [data, setData] = useState<AnalysisResponse | null>(null)
	const [error, setError] = useState<string | null>(null)
	const [loading, setLoading] = useState(true)

	useEffect(() => {
		async function run() {
			if (!id) return
			setLoading(true)
			setError(null)
			try {
				const res = await axios.get(`/api/analyze/${id}`)
				setData(res.data)
			} catch (e: any) {
				setError(e?.response?.data?.detail || e.message || 'Failed to load analysis')
			} finally {
				setLoading(false)
			}
		}
		run()
	}, [id])

	const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

	return (
		<div className="space-y-6">
			<h2 className="text-xl font-semibold">Results</h2>
			{loading && <p>Analyzing…</p>}
			{error && <p className="text-red-400">{error}</p>}
			{data && (
				<div className="space-y-4">
					<div className="text-sm opacity-80">
						<p>Duration: {data.duration_sec ? `${data.duration_sec.toFixed(2)}s` : '—'}</p>
						<p>Frames: {data.frame_count ?? '—'}</p>
					</div>
					<div>
						<h3 className="font-medium mb-2">Annotated Frames</h3>
						<div className="grid grid-cols-2 gap-4">
							{/* Support both backend variants */}
							{data.key_frames?.map((kf) => (
								<div key={`kf-${kf.frame_index}`} className="border border-slate-700 rounded overflow-hidden">
									<img src={kf.image_path} alt={`frame ${kf.frame_index}`} />
									<div className="p-2 text-xs">Frame {kf.frame_index}</div>
								</div>
							))}
							{!data.key_frames && data.annotations?.map((a) => (
								<div key={`ann-${a.frame_index}`} className="border border-slate-700 rounded overflow-hidden">
									{a.preview_image ? (
										<img src={`${backendUrl}/uploads/${a.preview_image}`} alt={`frame ${a.frame_index}`} />
									) : (
										<div className="p-4 text-xs">No image</div>
									)}
									<div className="p-2 text-xs">Frame {a.frame_index}</div>
								</div>
							))}
						</div>
					</div>
					<div>
						<h3 className="font-medium mb-2">Feedback</h3>
						<ul className="list-disc pl-5 space-y-1 text-sm">
							{data.feedback?.tips?.map((t, i) => (
								<li key={i}>{t}</li>
							)) || <li>No tips available</li>}
						</ul>
					</div>
				</div>
			)}
		</div>
	)
}
