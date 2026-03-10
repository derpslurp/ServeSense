import './globals.css'
import type { Metadata } from 'next'
import { ThemeProviderWrapper } from '../components/ThemeProviderWrapper'

export const metadata: Metadata = {
	title: 'ServeSense',
	description: 'Volleyball serve analysis',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en" suppressHydrationWarning>
			<body className="min-h-screen bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 transition-colors duration-200">
				<ThemeProviderWrapper>
					{children}
				</ThemeProviderWrapper>
			</body>
		</html>
	)
}
