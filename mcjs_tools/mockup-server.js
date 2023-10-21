const express = require('express')
const process = require('process')
const fs = require('fs')

const app = express()
const port = 3000

const mockupData = {
  breakpoints: [
    {
      filename: 'node_modules/json5/dist/index.mjs',
      line: 539,
    },

    {
      filename: 'node_modules/json5/dist/index.mjs',
      line: 539,
    },

    {
      filename: 'node_modules/json5/dist/index.mjs',
      line: 539,
    },

    {
      filename: 'test_parse.mjs',
      line: 17,
    },
  ],

  // bottom to top
  frames: [
    {
      viewMode: 'source',
      sourceFile: 'test_parse.mjs',
      sourceLine: 17,

      callID: 33,
      functionID: 2,
      thisValue: 'undefined', 
      returnToInstrID: 'undefined',

      numArgs: 0,
      captures: [],
      results: [
        5,
        345,
        12,
        'asd',
        'lol',
        4593,
        null,
        { type: 'object', objectId: 123 },
      ],
    },

    {
      viewMode: 'bytecode',
      sourceFile: 'node_modules/json5/dist/index.mjs',
      sourceLine: 539,

      callID: 34,
      functionID: 89,
      thisValue: 'undefined', 
      returnToInstrID: 93,

      numArgs: 3,
      captures: [
        {upvalueId: { index: 18, gen: 1 }},
        {upvalueId: { index: 27, gen: 1 }},
        {upvalueId: { index: 98, gen: 1 }},
      ],
      results: [
        5,
        345,
        12,
        'asd',
        'lol',
        4593,
        null,
        { type: 'object', objectId: 123 },
      ],
    },
  ],

  functions: {
    89: {
      bytecode: [
        'ObjCreateEmpty(v7)',
        'LoadConst(v8, const[3]) = (String("context"))',
        'LoadConst(v10, const[4]) = (String("Array"))',
        'GetGlobal { dest: v10, key: v10 }',
        'LoadUndefined(v11)',
        'Call { return_value: v9, this: v11, callee: v10 }',
        'LoadConst(v11, const[5]) = (String("prototype"))',
        'ObjGet { dest: v10, obj: v10, key: v11 }      -> Object(ObjectId(8v1))',
        'LoadConst(v11, const[6]) = (String("__proto__"))',
        'ObjSet { obj: v9, key: v11, value: v10 }',
        'ObjSet { obj: v7, key: v8, value: v9 }',
        'LoadConst(v12, const[7]) = (String("test"))',
        'ClosureNew { dest: v13, fnid: LocalFnId(2), forced_this: None }',
        'ObjSet { obj: v7, key: v12, value: v13 }',
        'LoadConst(v14, const[8]) = (String("strictSame"))',
        'ClosureNew { dest: v15, fnid: LocalFnId(3), forced_this: None }',
        'ObjSet { obj: v7, key: v14, value: v15 }',
        'LoadConst(v16, const[9]) = (String("equal"))',
        'ClosureNew { dest: v17, fnid: LocalFnId(4), forced_this: None }',
        'ObjSet { obj: v7, key: v16, value: v17 }',
        'LoadConst(v18, const[10]) = (String("ok"))',
        'ClosureNew { dest: v19, fnid: LocalFnId(5), forced_this: None }',
        'ObjSet { obj: v7, key: v18, value: v19 }',
        'LoadConst(v20, const[11]) = (String("end"))',
        'ClosureNew { dest: v21, fnid: LocalFnId(6), forced_this: None }',
        'ObjSet { obj: v7, key: v20, value: v21 }',
        'Copy { dst: v4, src: v7 }',
        'LoadConst(v22, const[12]) = (String("parse(text)"))',
        'LoadThis(v23)',
        'ClosureNew { dest: v24, fnid: LocalFnId(7), forced_this: Some(v23) }',
      ],
    },
  },

  objects: {
    123: {
      x: 1,
      y: 2,
    }
  },
}

const basePath = process.cwd() + '/../mcjs_vm/test-resources/test-scripts/json5/'
const filenames = [
  'test_parse.mjs',
  'node_modules/json5/dist/index.mjs',
]

mockupData.files = new Map()
for (const filename of filenames) {
  const fullPath = basePath + '/' + filename
  const content = fs.readFileSync(fullPath).toString()
  mockupData.files.set(filename, { content })
}

app.set('view engine', 'ejs')
app.set('views', './templates/')

app.use(express.static('data/'))

app.get('/', (_req, res) => {
  res.render('mockup', mockupData)
})

app.listen(port, () => {
  console.log(`Mockup web server available at localhost:${port}`)
})

