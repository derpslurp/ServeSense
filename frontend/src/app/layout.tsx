import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
	title: 'ServeSense',
	description: 'Volleyball serve analysis',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<body className="min-h-screen">
				<div className="max-w-3xl mx-auto p-6">
					<h1 className="text-2xl font-semibold mb-6">ServeSense</h1>
					{children}
				</div>
			</body>
		</html>
	)
}
