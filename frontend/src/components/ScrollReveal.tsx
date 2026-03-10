"use client";

import type { PropsWithChildren, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { useInView } from "../hooks/useInView";

type Direction = "up" | "down" | "left" | "right" | "none";

interface ScrollRevealProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  direction?: Direction;
  offset?: number;
  once?: boolean;
}

const directionTransforms: Record<Direction, string> = {
  up: "translateY(12px)",
  down: "translateY(-12px)",
  left: "translateX(12px)",
  right: "translateX(-12px)",
  none: "translateY(0)",
};

export default function ScrollReveal({
  children,
  className = "",
  delay = 0,
  direction = "up",
  offset = 12,
  once = true,
}: PropsWithChildren<ScrollRevealProps>) {
  const [ref, isInView] = useInView<HTMLDivElement>({ threshold: 0.1 });
  const [hasEntered, setHasEntered] = useState(false);

  useEffect(() => {
    if (isInView) setHasEntered(true);
  }, [isInView]);

  const transformFromDirection = useMemo(() => {
    if (direction === "none") return "translate3d(0,0,0)";
    const base = directionTransforms[direction] ?? "translateY(12px)";
    return base.replace("12", offset.toString());
  }, [direction, offset]);

  const isVisible = once ? hasEntered : isInView;

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "translate3d(0,0,0)" : transformFromDirection,
        transitionProperty: "opacity, transform",
        transitionDuration: "700ms",
        transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
        transitionDelay: `${delay}ms`,
        willChange: "opacity, transform",
      }}
    >
      {children}
    </div>
  );
}


