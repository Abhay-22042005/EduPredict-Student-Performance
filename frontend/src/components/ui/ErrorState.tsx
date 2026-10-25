import { AlertTriangle } from "lucide-react";
import { Button } from "./Button";

type ErrorStateProps = {
  message: string;
  onRetry?: () => void;
};

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-start gap-4 rounded-2xl border border-rose-300/20 bg-rose-300/[0.045] p-5" role="alert">
      <div className="flex items-start gap-3">
        <AlertTriangle aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-rose-200" />
        <div>
          <p className="font-semibold text-rose-100">Live data is unavailable</p>
          <p className="mt-1 text-sm leading-6 text-rose-100/70">{message}</p>
        </div>
      </div>
      {onRetry ? (
        <Button onClick={onRetry} type="button">
          Try again
        </Button>
      ) : null}
    </div>
  );
}
