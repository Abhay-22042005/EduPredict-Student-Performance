import type { HTMLAttributes, ReactNode } from "react";

type CardProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode;
  inset?: boolean;
};

export function Card({ children, className = "", inset = false, ...props }: CardProps) {
  return (
    <div
      className={`surface-panel rounded-2xl ${inset ? "p-4 sm:p-5" : "p-5 sm:p-6"} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
