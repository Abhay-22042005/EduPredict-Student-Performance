import { Component, lazy, Suspense, useMemo } from "react";
import type { ReactNode } from "react";
import {
  ArrowDownRight,
  ArrowRight,
  BrainCircuit,
  Database,
  Gauge,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { Link } from "react-router-dom";

const DecorativeDataScene = lazy(() =>
  import("../components/landing/DecorativeDataScene").then((module) => ({
    default: module.DecorativeDataScene,
  })),
);

const pipeline = [
  { label: "Data", icon: Database },
  { label: "Preprocessing", icon: Layers3 },
  { label: "Model", icon: BrainCircuit },
  { label: "Prediction", icon: Gauge },
];

function StaticSceneFallback() {
  return (
    <div aria-hidden="true" className="scene-static-fallback">
      <div className="scene-fallback-orbit scene-fallback-orbit-one" />
      <div className="scene-fallback-orbit scene-fallback-orbit-two" />
      <div className="scene-fallback-core" />
      <span className="scene-fallback-node scene-fallback-node-one" />
      <span className="scene-fallback-node scene-fallback-node-two" />
      <span className="scene-fallback-node scene-fallback-node-three" />
      <span className="scene-fallback-node scene-fallback-node-four" />
      <span className="scene-fallback-caption">A connected view of learning</span>
    </div>
  );
}

class SceneLoadBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("The decorative landing scene could not be loaded.", error);
  }

  render() {
    return this.state.hasError ? <StaticSceneFallback /> : this.props.children;
  }
}

function ScenePanel() {
  const sceneFallback = useMemo(() => <StaticSceneFallback />, []);

  return (
    <div aria-hidden="true" className="hero-scene-panel">
      <div className="hero-scene-wash" />
      <SceneLoadBoundary>
        <Suspense fallback={sceneFallback}>
          <DecorativeDataScene />
        </Suspense>
      </SceneLoadBoundary>
      <div className="hero-scene-caption">
        <span className="h-1.5 w-1.5 rounded-full bg-teal shadow-[0_0_10px_rgba(57,198,177,.8)]" />
        A connected view of learning
      </div>
      <div className="hero-scene-meta">
        <span>DATA NETWORK</span>
        <span>EDUPREDICT / 01</span>
      </div>
    </div>
  );
}

export function LandingPage() {
  const reduceMotion = useReducedMotion();
  const reveal = {
    initial: reduceMotion ? false : { opacity: 0, y: 16 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.2 },
    transition: { duration: reduceMotion ? 0 : 0.5, ease: "easeOut" as const },
  };

  return (
    <div className="space-y-16 sm:space-y-20">
      <section className="hero-shell">
        <div aria-hidden="true" className="soft-grid absolute inset-0 opacity-40" />
        <div aria-hidden="true" className="hero-ambient-glow" />
        <div className="hero-content-grid">
          <motion.div className="hero-copy" {...reveal}>
            <div className="inline-flex items-center gap-2 rounded-full border border-teal/20 bg-teal/[0.07] px-3 py-1.5 text-xs font-semibold text-teal">
              <span className="h-1.5 w-1.5 rounded-full bg-teal" />
              Student performance intelligence
            </div>
            <h1
              aria-label="Student Performance Intelligence"
              className="mt-7 max-w-2xl font-display text-4xl font-extrabold leading-[1.04] tracking-[-0.065em] text-white sm:text-5xl xl:text-[3.85rem]"
            >
              Student Performance
              <br />
              <span className="bg-gradient-to-r from-cyan via-teal to-violet bg-clip-text text-transparent">
                Intelligence.
              </span>
            </h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-slate-300 sm:text-lg sm:leading-8">
              Explore student-related data and estimate examination scores with a
              trained machine-learning model. Clear evidence, honest limitations, and
              the learner stay at the center.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link
                className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-teal/45 bg-teal/15 px-5 py-2.5 text-sm font-bold text-cyan transition-colors hover:border-cyan/60 hover:bg-teal/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan"
                to="/dashboard"
              >
                Explore dashboard
                <ArrowRight aria-hidden="true" className="h-4 w-4" />
              </Link>
              <Link
                className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-line bg-white/[0.035] px-5 py-2.5 text-sm font-semibold text-slate-200 transition-colors hover:bg-white/[0.07] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-cyan"
                to="/predict"
              >
                Try a prediction
                <ArrowDownRight aria-hidden="true" className="h-4 w-4" />
              </Link>
            </div>
            <div className="mt-7 flex items-start gap-3 border-l border-teal/40 pl-4">
              <ShieldCheck aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
              <p className="max-w-lg text-xs leading-5 text-muted">
                Predictions are estimates based on the available dataset, not
                guaranteed academic outcomes or judgments about a student.
              </p>
            </div>
          </motion.div>

          <motion.div
            className="hero-scene-column"
            initial={reduceMotion ? false : { opacity: 0, scale: 0.975 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: reduceMotion ? 0 : 0.75, delay: reduceMotion ? 0 : 0.08 }}
          >
            <ScenePanel />
          </motion.div>
        </div>

        <motion.div className="pipeline-strip" {...reveal}>
          <div className="pipeline-intro">
            <p className="eyebrow">A transparent pipeline</p>
            <p className="mt-1 text-xs text-muted">From observed inputs to an estimate</p>
          </div>
          <div className="pipeline-steps">
            {pipeline.map(({ label, icon: Icon }, index) => (
              <div className="pipeline-step" key={label}>
                <span className="pipeline-step-icon">
                  <Icon aria-hidden="true" className="h-4 w-4" />
                </span>
                <span className="text-xs font-semibold text-slate-200">{label}</span>
                {index < pipeline.length - 1 ? (
                  <span aria-hidden="true" className="pipeline-connector" />
                ) : null}
              </div>
            ))}
          </div>
          <p className="pipeline-model-note">
            Saved Ridge regression pipeline
            <span className="text-slate-600"> / </span>
            Exam_Score estimate
          </p>
        </motion.div>
      </section>

      <motion.section className="space-y-5" {...reveal}>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">A considered approach</p>
            <h2 className="mt-2 font-display text-2xl font-bold tracking-tight text-white">
              Clarity before complexity.
            </h2>
          </div>
          <p className="max-w-lg text-sm leading-6 text-muted">
            Explore patterns from the actual project dataset and saved model reports.
            Associations and feature importance do not establish causation.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            ["Grounded in observed data", "Dataset summaries and distributions come from the local API."],
            ["Measured, not promised", "Model evaluation is displayed from the saved training reports."],
            ["Designed for responsible use", "An estimate can support inquiry; it cannot define potential."],
          ].map(([title, description], index) => (
            <motion.article
              className="surface-panel h-full rounded-2xl p-5 sm:p-6"
              key={title}
              {...reveal}
              transition={{
                ...reveal.transition,
                delay: reduceMotion ? 0 : index * 0.07,
              }}
            >
              <span className="brand-mark">
                {index === 0 ? (
                  <Database aria-hidden="true" className="h-5 w-5" />
                ) : index === 1 ? (
                  <Gauge aria-hidden="true" className="h-5 w-5" />
                ) : (
                  <ShieldCheck aria-hidden="true" className="h-5 w-5" />
                )}
              </span>
              <h3 className="mt-5 font-semibold text-slate-100">{title}</h3>
              <p className="mt-2 text-sm leading-6 text-muted">{description}</p>
            </motion.article>
          ))}
        </div>
      </motion.section>
    </div>
  );
}
