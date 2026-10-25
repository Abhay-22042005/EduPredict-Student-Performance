import { LoaderCircle } from "lucide-react";

type LoadingStateProps = {
  label?: string;
};

export function LoadingState({ label = "Loading live data..." }: LoadingStateProps) {
  return (
    <div className="flex min-h-36 items-center justify-center gap-3 rounded-2xl border border-dashed border-line px-5 text-sm text-muted" role="status">
      <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin text-cyan motion-reduce:animate-none" />
      {label}
    </div>
  );
}
