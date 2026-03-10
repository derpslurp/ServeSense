import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function POST(req: Request) {
	try {
		const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
		const form = await req.formData()
		const imageToss = form.get('image_toss')
		const imageContact = form.get('image_contact')
		if (!(imageToss instanceof Blob) || !(imageContact instanceof Blob)) {
			return NextResponse.json(
				{ detail: 'Both image_toss and image_contact are required' },
				{ status: 400 }
			)
		}

		const proxyForm = new FormData()
		proxyForm.append('image_toss', imageToss, (imageToss as any).name || 'toss.png')
		proxyForm.append('image_contact', imageContact, (imageContact as any).name || 'contact.png')
		const ph = form.get('player_height_cm')
		if (ph != null && ph !== '') proxyForm.append('player_height_cm', ph.toString())
		const tct = form.get('toss_contact_time_s')
		if (tct != null && tct !== '') proxyForm.append('toss_contact_time_s', tct.toString())

		const res = await fetch(`${backendUrl}/analyze-frames`, {
			method: 'POST',
			body: proxyForm,
		})

		const data = await res.json().catch(() => ({}))
		return NextResponse.json(data, { status: res.status })
	} catch (e: any) {
		return NextResponse.json({ detail: e.message || 'Proxy error' }, { status: 500 })
	}
}
