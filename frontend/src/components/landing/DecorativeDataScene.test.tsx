import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DecorativeDataScene } from "./DecorativeDataScene";

vi.mock("@react-three/fiber", () => ({
  Canvas: ({ fallback }: { fallback: ReactNode }) => (
    <div data-testid="webgl-fallback">{fallback}</div>
  ),
  useFrame: () => undefined,
}));

vi.mock("@react-three/drei", () => ({
  Float: ({ children }: { children: ReactNode }) => <>{children}</>,
  OrbitControls: () => null,
  Sparkles: () => null,
}));

describe("DecorativeDataScene", () => {
  it("renders a static scene when WebGL is unavailable", () => {
    render(<DecorativeDataScene />);

    expect(screen.getByTestId("webgl-fallback")).toBeInTheDocument();
    expect(screen.getByText("A connected view of learning")).toBeInTheDocument();
  });
});
