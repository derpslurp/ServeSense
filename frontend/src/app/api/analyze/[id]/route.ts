import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function GET(_req: Request, { params }: { params: { id: string } }) {
	try {
		const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
		const res = await fetch(`${backendUrl}/analyze/${params.id}`, { method: 'GET' })
		const data = await res.json().catch(() => ({}))
		return NextResponse.json(data, { status: res.status })
	} catch (e: any) {
		return NextResponse.json({ detail: e.message || 'Proxy error' }, { status: 500 })
	}
}
