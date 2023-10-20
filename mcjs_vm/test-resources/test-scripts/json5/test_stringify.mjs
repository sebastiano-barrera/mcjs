import JSON5 from 'json5';

sink(JSON5.stringify(null));
sink(JSON5.stringify(123));
sink(JSON5.stringify(456.78));
sink(JSON5.stringify(true));
sink(JSON5.stringify(false));

