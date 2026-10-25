import type { ReactNode } from "react";

type StatusBadgeProps = {
  children: ReactNode;
  tone: "ready" | "warning" | "error" | "neutral";
};

const tones = {
  ready: "border-teal/25 bg-teal/[0.09] text-teal",
  warning: "border-amber-300/25 bg-amber-300/[0.08] text-amber-200",
  error: "border-rose-300/25 bg-rose-300/[0.08] text-rose-200",
  neutral: "border-slate-400/20 bg-slate-400/[0.06] text-slate-300",
};

export function StatusBadge({ children, tone }: StatusBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold ${tones[tone]}`}>
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
      {children}
    </span>
  );
}
