import mod from 'index.mjs';

sink(mod.stringify(null));
sink(mod.stringify(123));
sink(mod.stringify(456.78));
sink(mod.stringify(true));
sink(mod.stringify(false));

