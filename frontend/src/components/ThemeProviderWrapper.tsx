'use client'

import { ThemeProvider } from '../contexts/ThemeContext'
import { ThemeToggle } from './ThemeToggle'
import { AnimatedBackground } from './AnimatedBackground'

export function ThemeProviderWrapper({ children }: { children: React.ReactNode }) {
	return (
		<ThemeProvider>
			<AnimatedBackground />
			<div className="max-w-7xl mx-auto p-6 relative z-10">
				<div className="flex items-center justify-center mb-6 relative">
					<h1 className="text-7xl font-bold text-slate-900 dark:text-slate-100 tracking-wide font-dimis">ServeSense</h1>
					<div className="absolute right-0">
						<ThemeToggle />
					</div>
				</div>
				{children}
			</div>
		</ThemeProvider>
	)
}
