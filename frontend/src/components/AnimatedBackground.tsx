'use client'

export function AnimatedBackground() {
	return (
		<div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
			{/* Animated gradient orbs with floating animation - more dynamic */}
			<div className="absolute top-0 -left-1/4 w-96 h-96 bg-blue-500/30 dark:bg-blue-500/20 rounded-full blur-3xl animate-float"></div>
			<div className="absolute top-1/4 -right-1/4 w-96 h-96 bg-purple-500/30 dark:bg-purple-500/20 rounded-full blur-3xl animate-float-delay-1"></div>
			<div className="absolute bottom-0 left-1/4 w-96 h-96 bg-fuchsia-500/30 dark:bg-fuchsia-500/20 rounded-full blur-3xl animate-float-delay-2"></div>
			<div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-emerald-500/20 dark:bg-emerald-500/15 rounded-full blur-3xl animate-float-delay-3"></div>
			<div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-amber-500/15 dark:bg-amber-500/10 rounded-full blur-3xl animate-float-delay-4"></div>
			
			{/* Additional smaller orbs for more movement */}
			<div className="absolute top-3/4 left-1/3 w-64 h-64 bg-cyan-500/20 dark:bg-cyan-500/15 rounded-full blur-3xl animate-float" style={{ animationDelay: '5s', animationDuration: '16s' }}></div>
			<div className="absolute top-1/3 right-1/3 w-72 h-72 bg-pink-500/25 dark:bg-pink-500/15 rounded-full blur-3xl animate-float" style={{ animationDelay: '6s', animationDuration: '24s' }}></div>
			
			{/* Animated moving gradient overlay */}
			<div className="absolute inset-0 bg-gradient-to-br from-blue-600/5 via-purple-600/5 to-fuchsia-600/5 dark:from-blue-600/10 dark:via-purple-600/10 dark:to-fuchsia-600/10 animate-gradient-xy"></div>
			
			{/* Additional rotating gradient */}
			<div className="absolute inset-0 bg-gradient-to-tl from-emerald-600/5 via-blue-600/5 to-purple-600/5 dark:from-emerald-600/10 dark:via-blue-600/10 dark:to-purple-600/10 animate-gradient-xy" style={{ animationDuration: '25s', animationDirection: 'reverse' }}></div>
		</div>
	)
}
