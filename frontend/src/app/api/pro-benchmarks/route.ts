import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function POST(req: Request) {
	try {
		const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
		const body = await req.json()
		const res = await fetch(`${backendUrl}/pro-benchmarks`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		})
		const data = await res.json().catch(() => ({}))
		return NextResponse.json(data, { status: res.status })
	} catch (e: any) {
		return NextResponse.json({ detail: (e as Error).message || 'Proxy error' }, { status: 500 })
	}
}
