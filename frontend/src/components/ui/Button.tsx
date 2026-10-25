import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: "primary" | "secondary" | "quiet";
};

const variants = {
  primary:
    "border border-teal/50 bg-teal/15 text-cyan hover:border-cyan/60 hover:bg-teal/20",
  secondary:
    "border border-line bg-white/[0.035] text-slate-200 hover:border-cyan/30 hover:bg-white/[0.07]",
  quiet: "border border-transparent text-slate-300 hover:bg-white/[0.06]",
};

export function Button({
  children,
  className = "",
  variant = "secondary",
  ...props
}: ButtonProps) {
  return (
    <button
      className={`inline-flex min-h-10 items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
