import { CircleDashed } from "lucide-react";

type EmptyStateProps = {
  title: string;
  description: string;
};

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex min-h-44 flex-col items-center justify-center rounded-2xl border border-dashed border-line px-6 py-8 text-center">
      <CircleDashed aria-hidden="true" className="h-7 w-7 text-cyan/70" />
      <h2 className="mt-4 font-semibold text-slate-100">{title}</h2>
      <p className="mt-2 max-w-xl text-sm leading-6 text-muted">{description}</p>
    </div>
  );
}
