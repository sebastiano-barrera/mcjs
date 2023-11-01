/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html", "./templates/**/*.ejs"],
  theme: {
    extend: {},
    fontFamily: {
      'mono': ['Iosevka Extended', 'Source Code Pro', 'ui-monospace']
    }
  },
  plugins: [],
}

