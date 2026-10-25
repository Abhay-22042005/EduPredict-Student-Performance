/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07111f",
        "ink-soft": "#0b1727",
        panel: "#101e30",
        "panel-raised": "#14243a",
        line: "#20344a",
        muted: "#91a3b8",
        cyan: "#69e3e3",
        teal: "#39c6b1",
        violet: "#a68af8",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Manrope", "Inter", "ui-sans-serif", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 48px rgba(57, 198, 177, 0.12)",
        panel: "0 18px 54px rgba(0, 0, 0, 0.2)",
      },
    },
  },
  plugins: [],
};
