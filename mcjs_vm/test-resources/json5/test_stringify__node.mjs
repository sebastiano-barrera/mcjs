import mod from './json5/dist/index.mjs';

function sink(x) { console.log(x) }

sink(mod.stringify(null));
sink(mod.stringify(123));
sink(mod.stringify(456.78));
sink(mod.stringify(true));
sink(mod.stringify(false));

