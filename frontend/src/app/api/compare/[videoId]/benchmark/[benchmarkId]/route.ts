import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function POST(
  request: NextRequest,
  { params }: { params: { videoId: string; benchmarkId: string } }
) {
  try {
    const body = await request.json().catch(() => ({}))
    const res = await fetch(
      `${BACKEND_URL}/compare/${params.videoId}/benchmark/${params.benchmarkId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      }
    )
    if (!res.ok) {
      const error = await res.json()
      // Handle FastAPI validation errors (422) - detail is an array of validation errors
      let errorMessage = 'Comparison failed'
      if (error.detail) {
        if (Array.isArray(error.detail)) {
          // Pydantic validation errors
          errorMessage = error.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          // Regular error message
          errorMessage = error.detail
        }
      } else if (error.error) {
        errorMessage = error.error
      }
      return NextResponse.json(
        { error: errorMessage },
        { status: res.status }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to compare video' },
      { status: 500 }
    )
  }
}


