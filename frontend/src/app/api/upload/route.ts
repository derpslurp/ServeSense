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

		const proxyForm = new FormData()
		proxyForm.append('file', file, (file as any).name || 'upload.mp4')

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
