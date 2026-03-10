import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function POST(req: Request) {
	try {
		const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
		const form = await req.formData()
		const file = form.get('file')
		if (!(file instanceof Blob)) {
			return NextResponse.json({ detail: 'Missing file' }, { status: 400 })
		}

		const cameraAngle = form.get('camera_angle')
		const serveType = form.get('serve_type')
		const playerHeightCm = form.get('player_height_cm')

		const proxyForm = new FormData()
		proxyForm.append('file', file, (file as any).name || 'upload.mp4')
		if (cameraAngle) proxyForm.append('camera_angle', cameraAngle.toString())
		if (serveType) proxyForm.append('serve_type', serveType.toString())
		if (playerHeightCm != null && playerHeightCm !== '') {
			const cm = parseFloat(playerHeightCm.toString())
			if (Number.isFinite(cm) && cm > 0 && cm < 300) {
				proxyForm.append('player_height_cm', String(cm))
			}
		}

		const res = await fetch(`${backendUrl}/upload`, {
			method: 'POST',
			body: proxyForm,
		})

		const data = await res.json().catch(() => ({}))
		return NextResponse.json(data, { status: res.status })
	} catch (e: any) {
		return NextResponse.json({ detail: e.message || 'Proxy error' }, { status: 500 })
	}
}
