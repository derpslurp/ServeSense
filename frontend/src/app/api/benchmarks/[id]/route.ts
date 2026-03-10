import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const res = await fetch(`${BACKEND_URL}/benchmarks/${params.id}`)
    if (!res.ok) {
      return NextResponse.json(
        { error: 'Benchmark not found' },
        { status: 404 }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch benchmark' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const res = await fetch(`${BACKEND_URL}/benchmarks/${params.id}`, {
      method: 'DELETE',
    })
    if (!res.ok) {
      return NextResponse.json(
        { error: 'Benchmark not found' },
        { status: 404 }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to delete benchmark' },
      { status: 500 }
    )
  }
}


