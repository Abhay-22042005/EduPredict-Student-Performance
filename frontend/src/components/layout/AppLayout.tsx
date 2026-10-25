import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Header } from "./Header";
import { PageTransition } from "./PageTransition";
import { Sidebar } from "./Sidebar";

export function AppLayout() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="layout-root">
      <Sidebar />
      {menuOpen ? (
        <>
          <button
            aria-label="Close navigation overlay"
            className="mobile-overlay lg:hidden"
            onClick={() => setMenuOpen(false)}
            type="button"
          />
          <Sidebar mobile onClose={() => setMenuOpen(false)} onNavigate={() => setMenuOpen(false)} />
        </>
      ) : null}
      <div className="content-column">
        <Header onMenuClick={() => setMenuOpen(true)} />
        <main className="page-content">
          <PageTransition key={location.pathname}>
            <Outlet />
          </PageTransition>
        </main>
        <footer className="border-t border-line/70 px-5 py-4 text-center text-[11px] text-slate-500 sm:text-left lg:px-10">
          EduPredict <span aria-hidden="true">/</span> Estimates support inquiry; they do not
          define student potential.
        </footer>
      </div>
    </div>
  );
}
