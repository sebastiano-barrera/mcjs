// vim:et:ts=2
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html", "./templates/**/*.ejs"],
  safelist: [
    'bg-white',
    'text-black',
    'bg-red-700',
    'relative',
    'bottom-[2lh]',
    'bg-sky-800'
  ],
  theme: {
    extend: {},
    fontFamily: {
      'mono': ['Iosevka Extended', 'Source Code Pro', 'ui-monospace']
    }
  },
  plugins: [],
}

