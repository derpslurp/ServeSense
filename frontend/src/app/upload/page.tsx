"use client"

import { useState } from 'react'
import axios from 'axios'
import { useRouter } from 'next/navigation'

export default function UploadPage() {
	const router = useRouter()
	const [file, setFile] = useState<File | null>(null)
	const [error, setError] = useState<string | null>(null)
	const [loading, setLoading] = useState(false)
	const [progress, setProgress] = useState(0)

	async function onSubmit(e: React.FormEvent) {
		e.preventDefault()
		setError(null)
		if (!file) {
			setError('Please choose a video file')
			return
		}
		setLoading(true)
		setProgress(0)
		try {
			const form = new FormData()
			form.append('file', file)
			const res = await axios.post('/api/upload', form, {
				headers: { 'Content-Type': 'multipart/form-data' },
				onUploadProgress: (evt) => {
					if (evt.total) {
						setProgress(Math.round((evt.loaded / evt.total) * 100))
					}
				},
			})
			const videoId = res.data?.video_id || res.data?.videoId
			if (!videoId) throw new Error('Invalid response from server')
			router.push(`/results/${videoId}`)
		} catch (err: any) {
			setError(err?.response?.data?.detail || err.message || 'Upload failed')
		} finally {
			setLoading(false)
		}
	}

	return (
		<div>
			<h2 className="text-xl font-medium mb-4">Upload your serve video</h2>
			<form onSubmit={onSubmit} className="space-y-4">
				<input
					accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
					type="file"
					onChange={(e) => setFile(e.target.files?.[0] || null)}
					className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-slate-700 file:text-slate-100 hover:file:bg-slate-600"
				/>
				<button
					type="submit"
					disabled={loading}
					className="px-4 py-2 bg-blue-600 rounded text-white disabled:opacity-50"
				>
					{loading ? 'Uploading…' : 'Upload'}
				</button>
			</form>
			{loading && (
				<div className="mt-3">
					<div className="h-2 w-full bg-slate-700 rounded">
						<div className="h-2 bg-blue-500 rounded" style={{ width: `${progress}%` }} />
					</div>
					<p className="text-xs mt-1">{progress}%</p>
				</div>
			)}
			{error && <p className="text-red-400 mt-3">{error}</p>}
		</div>
	)
}
