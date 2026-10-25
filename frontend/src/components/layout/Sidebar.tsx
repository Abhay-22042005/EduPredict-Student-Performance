import {
  Activity,
  BookOpenCheck,
  BrainCircuit,
  ChartNoAxesCombined,
  CircleHelp,
  Compass,
  Gauge,
  GraduationCap,
  Sparkles,
  WandSparkles,
  X,
} from "lucide-react";
import { NavLink } from "react-router-dom";

type SidebarProps = {
  mobile?: boolean;
  onNavigate?: () => void;
  onClose?: () => void;
};

const navigation = [
  { label: "Home", path: "/", icon: Compass, end: true },
  { label: "Overview", path: "/dashboard", icon: Gauge },
  { label: "Analytics", path: "/analytics", icon: ChartNoAxesCombined },
  { label: "Predict", path: "/predict", icon: WandSparkles },
  { label: "Model lab", path: "/models", icon: BrainCircuit },
  { label: "Feature insights", path: "/insights", icon: Activity },
  { label: "Methodology", path: "/about", icon: CircleHelp },
];

export function Sidebar({ mobile = false, onNavigate, onClose }: SidebarProps) {
  return (
    <aside
      className={`${mobile ? "mobile-sidebar" : "desktop-sidebar"} px-4 py-5`}
    >
      <div className="flex items-center justify-between px-2">
        <NavLink
          aria-label="EduPredict home"
          className="flex items-center gap-3"
          onClick={onNavigate}
          to="/"
        >
          <span className="brand-mark">
            <GraduationCap aria-hidden="true" className="h-5 w-5" />
          </span>
          <span>
            <span className="block font-display text-sm font-extrabold tracking-tight text-white">
              EduPredict
            </span>
            <span className="mt-0.5 block font-mono text-[9px] uppercase tracking-[0.15em] text-muted">
              Intelligence suite
            </span>
          </span>
        </NavLink>
        {mobile ? (
          <button
            aria-label="Close navigation"
            className="rounded-lg p-2 text-muted hover:bg-white/[0.06] hover:text-white"
            onClick={onClose}
            type="button"
          >
            <X aria-hidden="true" className="h-5 w-5" />
          </button>
        ) : null}
      </div>

      <div className="mt-10 px-3">
        <p className="eyebrow text-[9px]">Workspace</p>
      </div>
      <nav aria-label="Main navigation" className="mt-3 flex flex-col gap-1">
        {navigation.map(({ label, path, icon: Icon, end }) => (
          <NavLink
            aria-label={label}
            className={({ isActive }) =>
              `nav-link px-3 py-2.5 text-[13px] font-semibold ${isActive ? "nav-link-active" : ""}`
            }
            end={end}
            key={path}
            onClick={onNavigate}
            to={path}
          >
            <Icon aria-hidden="true" className="h-[17px] w-[17px]" strokeWidth={1.8} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto pt-8">
        <div className="rounded-2xl border border-line bg-white/[0.025] p-4">
          <div className="flex items-center gap-2 text-cyan">
            <Sparkles aria-hidden="true" className="h-4 w-4" />
            <span className="text-xs font-semibold text-slate-100">Evidence first</span>
          </div>
          <p className="mt-2 text-xs leading-5 text-muted">
            Estimates are decision support, not guarantees or judgments about a student.
          </p>
          <div className="mt-4 flex items-center gap-2 border-t border-line pt-3 text-[10px] text-slate-500">
            <BookOpenCheck aria-hidden="true" className="h-3.5 w-3.5" />
            Model and data stay local
          </div>
        </div>
      </div>
    </aside>
  );
}
