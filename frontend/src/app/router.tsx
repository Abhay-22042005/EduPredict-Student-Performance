import { lazy, Suspense, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { AppLayout } from "../components/layout/AppLayout";
import { LoadingState } from "../components/ui/LoadingState";

const LandingPage = lazy(() =>
  import("../pages/LandingPage").then((module) => ({ default: module.LandingPage })),
);
const DashboardPage = lazy(() =>
  import("../pages/DashboardPage").then((module) => ({ default: module.DashboardPage })),
);
const AnalyticsPage = lazy(() =>
  import("../pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })),
);
const PredictionPage = lazy(() =>
  import("../pages/PredictionPage").then((module) => ({ default: module.PredictionPage })),
);
const ModelLabPage = lazy(() =>
  import("../pages/ModelLabPage").then((module) => ({ default: module.ModelLabPage })),
);
const InsightsPage = lazy(() =>
  import("../pages/InsightsPage").then((module) => ({ default: module.InsightsPage })),
);
const AboutPage = lazy(() =>
  import("../pages/AboutPage").then((module) => ({ default: module.AboutPage })),
);

function LazyPage({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<LoadingState label="Opening workspace..." />}>
      {children}
    </Suspense>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<LazyPage><LandingPage /></LazyPage>} />
        <Route path="dashboard" element={<LazyPage><DashboardPage /></LazyPage>} />
        <Route path="analytics" element={<LazyPage><AnalyticsPage /></LazyPage>} />
        <Route path="predict" element={<LazyPage><PredictionPage /></LazyPage>} />
        <Route path="models" element={<LazyPage><ModelLabPage /></LazyPage>} />
        <Route path="insights" element={<LazyPage><InsightsPage /></LazyPage>} />
        <Route path="about" element={<LazyPage><AboutPage /></LazyPage>} />
        <Route
          path="*"
          element={
            <main className="surface-panel rounded-2xl p-8">
              <p className="eyebrow">404 / Page not found</p>
              <h1 className="mt-3 text-2xl font-semibold text-white">
                This destination is not available.
              </h1>
              <p className="mt-2 text-sm text-muted">
                Use the navigation to return to a workspace page.
              </p>
            </main>
          }
        />
      </Route>
    </Routes>
  );
}
