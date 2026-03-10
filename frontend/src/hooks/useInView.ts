"use client"

import { useEffect, useRef, useState, MutableRefObject } from "react"

export function useInView<T extends HTMLElement>(
	options?: IntersectionObserverInit,
): [MutableRefObject<T | null>, boolean] {
	const elementRef = useRef<T | null>(null)
	const [isInView, setIsInView] = useState(false)

	useEffect(() => {
		const node = elementRef.current
		if (!node) return

		if (typeof window === "undefined" || typeof IntersectionObserver === "undefined") {
			setIsInView(true)
			return
		}

		const observer = new IntersectionObserver(
			(entries) => {
				const entry = entries[0]
				setIsInView(entry.isIntersecting)
			},
			{
				root: options?.root ?? null,
				rootMargin: options?.rootMargin ?? "0px",
				threshold: options?.threshold ?? 0.1,
			},
		)

		observer.observe(node)

		return () => {
			observer.unobserve(node)
			observer.disconnect()
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [
		elementRef,
		options?.root,
		options?.rootMargin,
		Array.isArray(options?.threshold) ? options?.threshold.join(",") : options?.threshold,
	])

	return [elementRef, isInView]
}



