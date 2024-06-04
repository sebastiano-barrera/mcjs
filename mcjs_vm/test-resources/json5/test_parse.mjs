/// This module is intended to be used as a test suite for mcjs, indirectly 
/// testing the correctness of the interpreter by testing the correctness of
/// the behavior of a module executed by the interpreter.
/// 
/// It is liberally based on json5's original "test/parse.js" [1].
///
/// [1] https://github.com/json5/json5/blob/main/test/parse.js

import JSON5 from 'json5';

// Instead of requiring 'tap', we're providing an implementation of its interface.
// *Just* enough to run this module, and nothing more.
const t = {
	context: [],
	test(description, initializer) {
		this.context.push({ description })
		initializer(this)
		this.context.pop({ description })
	},

	strictSame(found, expected, errorMessage) {
		if (found === expected) { return true; }
    if (typeof found !== typeof expected) { return false; }

    // NaN
    if (found !== found) { return expected !== expected; }

    if (typeof found === 'object') {
      for (const k in expected) {
        if (!this.strictSame(found[k], expected[k])) {
          return false;
        }
      }

      // extra keys in `found` are OK
      return true;
    }

    throw `assert failed: ${found} !== ${expected}: ${errorMessage}`;
	},
	equal(found, expected, errorMessage) { 
		// TODO Fix this when `==` is implemented differently from `===`
		return this.strictSame(found, expected, errorMessage)
	},
	ok(boolValue, errorMessage) {
		this.strictSame(boolValue, true, errorMessage)
	},

	// Do nothing
	end() {},
};


t.test('parse(text)', t => {
	t.test('objects', t => {
		t.strictSame(
			JSON5.parse('{}'),
			{},
			'parses empty objects'
		)

		t.strictSame(
			JSON5.parse('{"a":1}'),
			{ a: 1 },
			'parses double string property names'
		)

		t.strictSame(
			JSON5.parse("{'a':1}"),
			{ a: 1 },
			'parses single string property names'
		)

		t.strictSame(
			JSON5.parse('{a:1}'),
			{ a: 1 },
			'parses unquoted property names'
		)

		t.strictSame(
			JSON5.parse('{$_:1,_$:2,a\u200C:3}'),
			{ $_: 1, _$: 2, 'a\u200C': 3 },
			'parses special character property names'
		)

		t.strictSame(
			JSON5.parse('{ùńîċõďë:9}'),
			{ 'ùńîċõďë': 9 },
			'parses unicode property names'
		)

		t.strictSame(
			JSON5.parse('{\\u0061\\u0062:1,\\u0024\\u005F:2,\\u005F\\u0024:3}'),
			{ ab: 1, $_: 2, _$: 3 },
			'parses escaped property names'
		)

		t.strictSame(
			// eslint-disable-next-line no-proto
			JSON5.parse('{"__proto__":1}').__proto__,
			1,
			'preserves __proto__ property names'
		)

		t.strictSame(
			JSON5.parse('{abc:1,def:2}'),
			{ abc: 1, def: 2 },
			'parses multiple properties'
		)

		t.strictSame(
			JSON5.parse('{a:{b:2}}'),
			{ a: { b: 2 } },
			'parses nested objects'
		)

		t.end()
	})

	t.test('arrays', t => {
		t.strictSame(
			JSON5.parse('[]'),
			[],
			'parses empty arrays'
		)

		t.strictSame(
			JSON5.parse('[1]'),
			[1],
			'parses array values'
		)

		t.strictSame(
			JSON5.parse('[1,2]'),
			[1, 2],
			'parses multiple array values'
		)

		t.strictSame(
			JSON5.parse('[1,[2,3]]'),
			[1, [2, 3]],
			'parses nested arrays'
		)

		t.end()
	})

	t.test('nulls', t => {
		t.equal(
			JSON5.parse('null'),
			null,
			'parses nulls'
		)

		t.end()
	})

	t.test('Booleans', t => {
		t.equal(
			JSON5.parse('true'),
			true,
			'parses true'
		)

		t.equal(
			JSON5.parse('false'),
			false,
			'parses false'
		)

		t.end()
	})

	t.test('numbers', t => {
		t.strictSame(
			JSON5.parse('[0,0.,0e0]'),
			[0, 0, 0],
			'parses leading zeroes'
		)

		t.strictSame(
			JSON5.parse('[1,23,456,7890]'),
			[1, 23, 456, 7890],
			'parses integers'
		)

		t.strictSame(
			JSON5.parse('[-1,+2,-.1,-0]'),
			[-1, +2, -0.1, -0],
			'parses signed numbers'
		)

		t.strictSame(
			JSON5.parse('[.1,.23]'),
			[0.1, 0.23],
			'parses leading decimal points'
		)

		t.strictSame(
			JSON5.parse('[1.0,1.23]'),
			[1, 1.23],
			'parses fractional numbers'
		)

		t.strictSame(
			JSON5.parse('[1e0,1e1,1e01,1.e0,1.1e0,1e-1,1e+1]'),
			[1, 10, 10, 1, 1.1, 0.1, 10],
			'parses exponents'
		)

		t.strictSame(
			JSON5.parse('[0x1,0x10,0xff,0xFF]'),
			[1, 16, 255, 255],
			'parses hexadecimal numbers'
		)

		t.strictSame(
			JSON5.parse('[Infinity,-Infinity]'),
			[Infinity, -Infinity],
			'parses signed and unsigned Infinity'
		)

		t.ok(
			isNaN(JSON5.parse('NaN')),
			'parses NaN'
		)

		t.ok(
			isNaN(JSON5.parse('-NaN')),
			'parses signed NaN'
		)

		t.strictSame(
			JSON5.parse('1'),
			1,
			'parses 1'
		)

		t.strictSame(
			JSON5.parse('+1.23e100'),
			1.23e100,
			'parses +1.23e100'
		)

		t.strictSame(
			JSON5.parse('0x1'),
			0x1,
			'parses bare hexadecimal number'
		)

		t.strictSame(
			JSON5.parse('-0x0123456789abcdefABCDEF'),
			-0x0123456789abcdefABCDEF,
			'parses bare long hexadecimal number'
		)

		t.end()
	})

	t.test('strings', t => {
		t.equal(
			JSON5.parse('"abc"'),
			'abc',
			'parses double quoted strings'
		)

		t.equal(
			JSON5.parse("'abc'"),
			'abc',
			'parses single quoted strings'
		)

		t.strictSame(
			JSON5.parse(`['"',"'"]`),
			['"', "'"],
			'parses quotes in strings')

		t.equal(
			JSON5.parse(`'\\b\\f\\n\\r\\t\\v\\0\\x0f\\u01fF\\\n\\\r\n\\\r\\\u2028\\\u2029\\a\\'\\"'`),
			`\b\f\n\r\t\v\0\x0f\u01FF\a'"`, // eslint-disable-line no-useless-escape
			'parses escaped characters'
		)

		t.end()
	})

	t.test('comments', t => {
		t.strictSame(
			JSON5.parse('{//comment\n}'),
			{},
			'parses single-line comments'
		)

		t.strictSame(
			JSON5.parse('{}//comment'),
			{},
			'parses single-line comments at end of input'
		)

		t.strictSame(
			JSON5.parse('{/*comment\n** */}'),
			{},
			'parses multi-line comments'
		)

		t.end()
	})

	t.test('whitespace', t => {
		t.strictSame(
			JSON5.parse('{\t\v\f \u00A0\uFEFF\n\r\u2028\u2029\u2003}'),
			{},
			'parses whitespace'
		)

		t.end()
	})

	t.end()
})

t.test('parse(text, reviver)', t => {
	t.strictSame(
		JSON5.parse('{a:1,b:2}', (k, v) => (k === 'a') ? 'revived' : v),
		{ a: 'revived', b: 2 },
		'modifies property values'
	)

	t.strictSame(
		JSON5.parse('{a:{b:2}}', (k, v) => (k === 'b') ? 'revived' : v),
		{ a: { b: 'revived' } },
		'modifies nested object property values'
	)

	t.strictSame(
		JSON5.parse('{a:1,b:2}', (k, v) => (k === 'a') ? undefined : v),
		{ b: 2 },
		'deletes property values'
	)

	t.strictSame(
		JSON5.parse('[0,1,2]', (k, v) => (k === '1') ? 'revived' : v),
		[0, 'revived', 2],
		'modifies array values'
	)

	t.strictSame(
		JSON5.parse('[0,[1,2,3]]', (k, v) => (k === '2') ? 'revived' : v),
		[0, [1, 2, 'revived']],
		'modifies nested array values'
	)

	t.strictSame(
		JSON5.parse('[0,1,2]', (k, v) => (k === '1') ? undefined : v),
		[0, , 2], // eslint-disable-line no-sparse-arrays
		'deletes array values'
	)

	t.equal(
		JSON5.parse('1', (k, v) => (k === '') ? 'revived' : v),
		'revived',
		'modifies the root value'
	)

	t.strictSame(
		JSON5.parse('{a:{b:2}}', function(k, v) { return (k === 'b' && this.b) ? 'revived' : v }),
		{ a: { b: 'revived' } },
		'sets `this` to the parent value'
	)

	t.end()
})
