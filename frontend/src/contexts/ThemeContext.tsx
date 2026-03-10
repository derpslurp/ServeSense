'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'light' | 'dark'

interface ThemeContextType {
	theme: Theme
	toggleTheme: () => void
	setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
	const [theme, setThemeState] = useState<Theme>('dark')
	const [mounted, setMounted] = useState(false)

	const setTheme = (newTheme: Theme) => {
		setThemeState(newTheme)
		if (typeof window !== 'undefined') {
			localStorage.setItem('theme', newTheme)
			document.documentElement.classList.toggle('dark', newTheme === 'dark')
		}
	}

	const toggleTheme = () => {
		const newTheme = theme === 'dark' ? 'light' : 'dark'
		setTheme(newTheme)
	}

	useEffect(() => {
		setMounted(true)
		// Check localStorage for saved theme preference
		if (typeof window !== 'undefined') {
			const savedTheme = localStorage.getItem('theme') as Theme | null
			if (savedTheme === 'light' || savedTheme === 'dark') {
				setThemeState(savedTheme)
				document.documentElement.classList.toggle('dark', savedTheme === 'dark')
			} else {
				// Default to system preference
				const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
				const initialTheme = prefersDark ? 'dark' : 'light'
				setThemeState(initialTheme)
				document.documentElement.classList.toggle('dark', initialTheme === 'dark')
			}
		}
	}, [])

	// Always provide the context, even before mounting (to prevent errors)
	// Use default theme until mounted and actual theme is loaded
	const contextValue = {
		theme,
		toggleTheme,
		setTheme,
	}

	return (
		<ThemeContext.Provider value={contextValue}>
			{children}
		</ThemeContext.Provider>
	)
}

export function useTheme() {
	const context = useContext(ThemeContext)
	if (context === undefined) {
		throw new Error('useTheme must be used within a ThemeProvider')
	}
	return context
}

