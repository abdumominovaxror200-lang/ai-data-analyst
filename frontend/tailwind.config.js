/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0e17",
          900: "#0f1420",
          850: "#131a29",
          800: "#182233",
          700: "#212d42",
          600: "#2c3a54",
          500: "#3c4d6b",
          400: "#5b6d8c",
          300: "#8494ae",
          200: "#b4bfd1",
          100: "#dde3ec",
          50: "#f4f6fa",
        },
        accent: {
          DEFAULT: "#5b8def",
          soft: "#8fb2f5",
          dim: "#324b7a",
        },
        good: "#3fbf8f",
        warn: "#e0a63c",
        bad: "#e0616f",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px -8px rgba(0,0,0,0.5)",
      },
      borderRadius: {
        xl2: "1rem",
      },
    },
  },
  plugins: [],
};
