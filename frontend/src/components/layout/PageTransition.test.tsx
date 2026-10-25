import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PageTransition } from "./PageTransition";

vi.mock("framer-motion", async () => {
  const React = await import("react");
  return {
    useReducedMotion: () => true,
    motion: {
      div: ({
        children,
        transition,
      }: {
        children: React.ReactNode;
        transition: { duration: number };
      }) =>
        React.createElement(
          "div",
          { "data-transition-duration": transition.duration },
          children,
        ),
    },
  };
});

describe("PageTransition", () => {
  it("disables animated duration when reduced motion is requested", () => {
    render(
      <PageTransition>
        <p>Accessible transition content</p>
      </PageTransition>,
    );

    expect(screen.getByText("Accessible transition content").parentElement).toHaveAttribute(
      "data-transition-duration",
      "0",
    );
  });
});
