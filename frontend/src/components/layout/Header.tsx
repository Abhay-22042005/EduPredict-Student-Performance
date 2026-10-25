import { Menu, PanelTop } from "lucide-react";
import { useLocation } from "react-router-dom";
import { BackendStatus } from "./BackendStatus";
import { Button } from "../ui/Button";

const titles: Record<string, { title: string; section: string }> = {
  "/": { title: "Welcome", section: "Student performance intelligence" },
  "/dashboard": { title: "Overview", section: "Your performance workspace" },
  "/analytics": { title: "Analytics", section: "Explore observed dataset patterns" },
  "/predict": { title: "Prediction", section: "Responsible score estimation" },
  "/models": { title: "Model lab", section: "Evaluation and model selection" },
  "/insights": { title: "Feature insights", section: "Model interpretation" },
  "/about": { title: "Methodology", section: "Data, approach, and limitations" },
};

type HeaderProps = {
  onMenuClick: () => void;
};

export function Header({ onMenuClick }: HeaderProps) {
  const location = useLocation();
  const page = titles[location.pathname] ?? {
    title: "Workspace",
    section: "Student performance intelligence",
  };

  return (
    <header className="sticky top-0 z-30 flex min-h-[4.8rem] items-center justify-between gap-4 border-b border-line bg-ink/75 px-4 backdrop-blur-xl sm:px-6 lg:px-10">
      <div className="flex min-w-0 items-center gap-3">
        <Button
          aria-label="Open navigation"
          className="h-10 w-10 px-0 lg:hidden"
          onClick={onMenuClick}
          type="button"
        >
          <Menu aria-hidden="true" className="h-5 w-5" />
        </Button>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <PanelTop aria-hidden="true" className="hidden h-3.5 w-3.5 text-teal sm:block" />
            <p className="truncate text-sm font-semibold text-slate-100">{page.title}</p>
          </div>
          <p className="mt-0.5 hidden truncate text-[11px] text-muted sm:block">
            {page.section}
          </p>
        </div>
      </div>
      <BackendStatus />
    </header>
  );
}
