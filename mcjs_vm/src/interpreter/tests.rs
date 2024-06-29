#![cfg(test)]

use std::path::PathBuf;

use super::*;
use crate::{bytecode::Literal, bytecode_compiler};

fn quick_run_script(code: &str) -> FinishedData {
    let mut loader = loader::Loader::new_cwd();
    let chunk_fnid = loader
        .load_script_anon(code.to_string())
        .expect("couldn't compile test script");
    complete_run(&mut loader, chunk_fnid)
}

fn complete_run(loader: &mut crate::Loader, root_fnid: FnID) -> FinishedData {
    let mut realm = Realm::new(loader);
    let vm = Interpreter::new(&mut realm, loader, root_fnid);
    match vm.run() {
        Ok(exit) => exit.expect_finished(),
        Err(err_box) => {
            let error = err_box.error.with_loader(&loader);
            panic!("{:?}", error);
        }
    }
}

#[test]
fn test_simple_call() {
    let output = quick_run_script("/* Here is some simple code: */ sink(1 + 4 + 99); ");
    assert_eq!(&[Some(Literal::Number(104.0))], &output.sink[..]);
}

#[test]
fn test_multiple_calls() {
    let output = quick_run_script("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ");
    assert_eq!(
        &[
            Some(Literal::Number(12. * 5.)),
            Some(Literal::Number(99. - 15.))
        ],
        &output.sink[..]
    );
}

#[test]
fn test_if() {
    let output = quick_run_script(
        "
            const x = 123;
            let y = 'a';
            if (x < 200) {
                y = 'b';
            }
            sink(y);
            ",
    );

    assert_eq!(&[Some(Literal::String("b".to_owned()))], &output.sink[..]);
}

#[test]
fn test_simple_fn() {
    let output = quick_run_script(
        "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
    );

    assert_eq!(&[Some(Literal::Number(3.0))], &output.sink[..]);
}

#[test]
fn test_fn_with_branch() {
    let output = quick_run_script(
        "
            function foo(mode, a, b) {
                if (mode === 'sum')
                    return a + b;
                else if (mode === 'product')
                    return a * b;
                else
                    return 'something else';
            }

            sink(foo('product', 9, 8));
            ",
    );

    assert_eq!(&[Some(Literal::Number(9.0 * 8.0))], &output.sink[..]);
}

#[test]
fn test_while() {
    let output = quick_run_script(
        "
            function sum_range(n) {
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    i++;
                }
                return ret;
            }

            sink(sum_range(4));
            ",
    );

    assert_eq!(&[Some(Literal::Number(10.0))], &output.sink[..]);
}

#[test]
fn test_switch() {
    let output = quick_run_script(
        "
            function trySwitch(x) {
                switch(x) {
                case 'a':
                case 'b':
                    sink(1);
                case 'c':
                    sink(2);
                    break;

                case 'd':
                    sink(3);
                }
                sink(99);
            }

            trySwitch('b');
            trySwitch('d');
            trySwitch('c');
            trySwitch('y');
            trySwitch('a');
            ",
    );
    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Number(1.0)),
            Some(Literal::Number(2.0)),
            Some(Literal::Number(99.0)),
            //
            Some(Literal::Number(3.0)),
            Some(Literal::Number(99.0)),
            //
            Some(Literal::Number(2.0)),
            Some(Literal::Number(99.0)),
            //
            Some(Literal::Number(99.0)),
            //
            Some(Literal::Number(1.0)),
            Some(Literal::Number(2.0)),
            Some(Literal::Number(99.0)),
        ]
    );
}

fn try_casting_bool(code: &str, expected_value: bool) {
    let output = quick_run_script(code);
    assert_eq!(&[Some(Literal::Bool(expected_value))], &output.sink[..]);
}
#[test]
fn test_boolean_cast_from_number() {
    try_casting_bool("sink(!123.994);", false);
    try_casting_bool("sink(!-123.994);", false);
    try_casting_bool("sink(!0.0);", true);
}
#[test]
fn test_boolean_cast_from_string() {
    try_casting_bool("sink(!'');", true);
    try_casting_bool("sink(!'asdlol');", false);
}
#[test]
fn test_boolean_cast_from_bool() {
    try_casting_bool("sink(!false);", true);
    try_casting_bool("sink(!true);", false);
}
#[test]
fn test_boolean_cast_from_object() {
    try_casting_bool("sink(!{a: 1, b: 2});", false);
    try_casting_bool("sink(!{});", false);
}
#[test]
fn test_boolean_cast_from_null() {
    try_casting_bool("sink(!null);", true);
}
#[test]
fn test_boolean_cast_from_undefined() {
    try_casting_bool("sink(!undefined);", true);
}
#[test]
fn test_boolean_cast_from_function() {
    try_casting_bool(
        "
            function my_fun() { }
            sink(!my_fun);
            ",
        false,
    );
}

#[test]
fn test_number_cast() {
    let output = quick_run_script(
        "
            sink(Number(123.0));
            sink(Number('-123.45e9'));
            sink(Number(true));
            sink(Number(false));
            sink(Number(null));
            sink(Number({a: 3}));
            sink(Number());
            ",
    );

    assert!(matches!(
        &output.sink[..],
        &[
            Some(Literal::Number(123.0)),
            Some(Literal::Number(-123450000000.0)),
            Some(Literal::Number(1.0)),
            Some(Literal::Number(0.0)),
            Some(Literal::Number(0.0)),
            Some(Literal::Number(a)),
            Some(Literal::Number(b)),
        ]
        if a.is_nan() && b.is_nan()
    ))
}

// Un-ignore when Symbol.for is implemented
#[test]
#[ignore]
fn test_number_cast_symbol() {
    let output = quick_run_script(
        "
            try { Number(Symbol.for('asd')) }
            catch (err) { sink(err.name) }
            ",
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::String("TypeError".to_string())),]
    )
}
#[test]
fn test_capture() {
    let output = quick_run_script(
        "
            // wrapping into iife makes sure that the shared variable is not a global
            (function() {
                let counter = 0;

                function f() {
                    function g() {
                        counter++;
                    }
                    g();
                    g();
                    sink(counter);
                }

                f();
                f();
                f();
                counter -= 5;
                sink(counter);
            })();
            ",
    );

    assert_eq!(
        &[
            Some(Literal::Number(2.0)),
            Some(Literal::Number(4.0)),
            Some(Literal::Number(6.0)),
            Some(Literal::Number(1.0))
        ],
        &output.sink[..]
    );
}

#[test]
fn test_object_init() {
    let output = quick_run_script(
        "
            const obj = {
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            }

            sink(obj.aString)
            sink(obj.aNumber)
            sink(obj.anotherObject.x)
            sink(obj.anotherObject.y)
            sink(obj.aFunction())
            ",
    );

    assert_eq!(5, output.sink.len());
    assert_eq!(&Some(Literal::String("asdlol123".into())), &output.sink[0]);
    assert_eq!(&Some(Literal::Number(1239423.4518923)), &output.sink[1]);
    assert_eq!(&Some(Literal::Number(123.0)), &output.sink[2]);
    assert_eq!(&Some(Literal::Number(899.0)), &output.sink[3]);
    assert_eq!(&Some(Literal::Number(42.0)), &output.sink[4]);
}

#[test]
fn test_typeof() {
    let output = quick_run_script(
        "
            let anObj = {}

            sink(typeof undefined)
            sink(typeof anObj.aNonExistantProperty)

            sink(typeof null)
            sink(typeof {})
            sink(typeof anObj)

            sink(typeof true)
            sink(typeof false)

            sink(typeof 123.0)
            sink(typeof -99.2)
            sink(typeof (156.0/0))
            sink(typeof (-156.0/0))
            sink(typeof (0/0))

            sink(typeof '')
            sink(typeof 'a string')

            sink(typeof (function() {}))
            ",
    );

    assert_eq!(
        &output.sink[..],
        &[
            Some(Literal::String("undefined".into())),
            Some(Literal::String("undefined".into())),
            Some(Literal::String("object".into())),
            Some(Literal::String("object".into())),
            Some(Literal::String("object".into())),
            Some(Literal::String("boolean".into())),
            Some(Literal::String("boolean".into())),
            Some(Literal::String("number".into())),
            Some(Literal::String("number".into())),
            Some(Literal::String("number".into())),
            Some(Literal::String("number".into())),
            Some(Literal::String("number".into())),
            Some(Literal::String("string".into())),
            Some(Literal::String("string".into())),
            Some(Literal::String("function".into())),
            // TODO(feat) BigInt (typeof -> "bigint")
            // TODO(feat) Symbol (typeof -> "symbol")
        ]
    );
}

#[test]
fn test_object_member_set() {
    let output = quick_run_script(
        "
            const pt = { x: 123, y: 4 }

            sink(pt.x)
            sink(pt.y)
            pt.y = 999
            sink(pt.x)
            sink(pt.y)
            ",
    );

    assert_eq!(4, output.sink.len());
    assert_eq!(&Some(Literal::Number(123.0)), &output.sink[0]);
    assert_eq!(&Some(Literal::Number(4.0)), &output.sink[1]);
    assert_eq!(&Some(Literal::Number(123.0)), &output.sink[2]);
    assert_eq!(&Some(Literal::Number(999.0)), &output.sink[3]);
}

#[test]
fn test_object_prototype() {
    let output = quick_run_script(
        "
            const a = { count: 99, name: 'lol', pos: {x: 32, y: 99} }
            const b = { name: 'another name' }
            b.__proto__ = a
            const c = { __proto__: b, count: 0 }

            sink(c.pos.y)
            sink(c.pos.x)
            c.pos.x = 12304
            sink(b.pos.x)
            sink(c.count)
            sink(c.name)
            b.name = 'another name yet'
            sink(c.name)
            ",
    );

    assert_eq!(
        &output.sink[..],
        &[
            Some(Literal::Number(99.0)),
            Some(Literal::Number(32.0)),
            Some(Literal::Number(12304.0)),
            Some(Literal::Number(0.0)),
            Some(Literal::String("another name".into())),
            Some(Literal::String("another name yet".into())),
        ]
    );
}

#[test]
fn test_for_in() {
    // TODO(small feat) This syntax is not yet implemented
    let output = quick_run_script(
        "
            const obj = {
                x: 12.0,
                y: 90.2,
                name: 'THE SPOT',
            };

            for (const name in obj) sink(name);
            ",
    );

    let mut sink: Vec<_> = output
        .sink
        .into_iter()
        .map(|value| match value {
            Some(Literal::String(s)) => s.clone(),
            other => panic!("not a String: {:?}", other),
        })
        .collect();
    sink.sort();
    assert_eq!(&sink[..], &["name", "x", "y"]);
}

#[test]
fn test_builtin() {
    let output = quick_run_script(
        "
            sink(Array.isArray([1, 2, 3]));
            sink(Array.isArray('not an array'));
            ",
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::Bool(true)), Some(Literal::Bool(false))]
    );
}

#[test]
fn test_make_exception() {
    #![allow(non_snake_case)]

    let mut loader = loader::Loader::new_cwd();
    let mut realm = Realm::new(&mut loader);
    let global_this = realm.global_obj();

    let message = "some message";
    let cons_name = "TypeError";
    let TypeError = realm.heap.get_chained_key(global_this, cons_name).unwrap();

    let exc = make_exception(&mut realm, cons_name, message);

    {
        let chk_message = realm.heap.get_chained_key(exc, "message").unwrap();
        let chk_message = realm.heap.as_str(chk_message).unwrap().to_string();
        assert_eq!(&chk_message, message);
    }

    {
        let chk_prototype = Value::Object(realm.heap.proto(exc).unwrap());
        let exp_prototype = realm.heap.get_chained_key(TypeError, "prototype").unwrap();
        assert_eq!(chk_prototype, exp_prototype);
    }

    {
        let chk_cons = realm.heap.get_chained_key(exc, "constructor").unwrap();
        assert_eq!(chk_cons, TypeError);
    }
}

#[test]
fn test_this_basic_nonstrict() {
    let output = quick_run_script(
        r#"
            function getThis() { return this; }
            sink(getThis());
            "#,
    );
    // `None` because it's the globalThis object due to "this substitution"
    assert_eq!(&output.sink, &[None]);
}

#[test]
fn test_this_basic_strict() {
    let output = quick_run_script(
        r#"
            "use strict";
            function getThis() { return this; }
            sink(getThis());
            "#,
    );
    // no "this substitution" in strict mode
    assert_eq!(&output.sink, &[Some(Literal::Undefined)]);
}

#[test]
fn test_this() {
    let output = quick_run_script(
        r#"
            "use strict"
            
            function getThis() {
                return this;
            }
            function getThisViaArrowFunc() {
                const f = () => { return this; };
                return f();
            }

            const obj1 = { name: "obj1" };
            const obj2 = { name: "obj2" };
            obj1.getThis = getThis;
            obj2.getThis = getThis;
            const obj3 = { __proto__: obj1, name: "obj3" };
            const obj4 = {
              name: "obj4",
              getThis() { return this },
            };
            const obj5 = { name: "obj5" };
            obj5.getThis = obj4.getThis;

            sink(obj1.getThis().name);
            sink(obj2.getThis().name);
            sink(obj3.getThis().name);
            sink(obj5.getThis().name);
            sink(getThis());

            obj5.getThisViaArrowFunc = getThisViaArrowFunc;
            sink(obj5.getThisViaArrowFunc().name);
            "#,
    );
    assert_eq!(
        &output.sink,
        &[
            Some(Literal::String("obj1".into())),
            Some(Literal::String("obj2".into())),
            Some(Literal::String("obj3".into())),
            Some(Literal::String("obj5".into())),
            Some(Literal::Undefined),
            Some(Literal::String("obj5".into())),
        ],
    );
}

#[test]
fn test_number_constructor_prototype() {
    let mut loader = loader::Loader::new_cwd();
    let realm = Realm::new(&mut loader);
    let number = realm
        .heap
        .get_chained(realm.global_obj, heap::IndexOrKey::Key("Number"))
        .unwrap()
        .value()
        .unwrap();
    let prototype = realm
        .heap
        .get_chained(number, heap::IndexOrKey::Key("prototype"))
        .unwrap()
        .value()
        .unwrap();
    assert_eq!(prototype, Value::Object(realm.heap.number_proto()));
}

#[test]
fn test_methods_on_numbers() {
    let output = quick_run_script(
        r#"
            const num = 123.45;

            Number.prototype.greet = function() { return "Hello, I'm " + this.toString() + "!" }

            sink(num.greet())
            "#,
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::String("Hello, I'm 123.45!".into())),],
    );
}

#[test]
fn test_array_init() {
    let output = quick_run_script("sink([].length)");
    assert_eq!(&output.sink, &[Some(Literal::Number(0.0))]);
}

#[test]
fn test_array_access() {
    let output = quick_run_script(
        r#"
            const xs = ['a', 'b', 'c'];

            sink(xs[-1])
            sink(xs[0])
            sink(xs[1])
            sink(xs[2])
            sink(xs[3])
            sink(xs.length)
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Undefined),
            Some(Literal::String("a".to_string())),
            Some(Literal::String("b".to_string())),
            Some(Literal::String("c".to_string())),
            Some(Literal::Undefined),
            Some(Literal::Number(3.0)),
        ],
    );
}

#[test]
fn test_script_global() {
    let output = quick_run_script(
        r#"
            var x = 55
            sink(globalThis.x)
            sink(x)

            globalThis.x = 222
            sink(x)
            sink(globalThis.x)
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Number(55.0)),
            Some(Literal::Number(55.0)),
            Some(Literal::Number(222.0)),
            Some(Literal::Number(222.0)),
        ],
    );
}

#[test]
fn test_script_global_fn_nonstrict() {
    let output = quick_run_script(
        r#"
            function x() { return 55 }
            sink(globalThis.x())
            sink(x())
            "#,
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::Number(55.0)), Some(Literal::Number(55.0)),],
    );
}

#[test]
fn test_script_global_fn_strict() {
    let output = quick_run_script(
        r#"
            "use strict";
            function x() { return 55 }
            sink(globalThis.x())
            sink(x())
            "#,
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::Number(55.0)), Some(Literal::Number(55.0)),],
    );
}

#[test]
fn test_constructor_prototype() {
    quick_run_script(
        r#"
                function Test262Error(message) {
                  this.message = message || "";
                }

                Test262Error.prototype.toString = function () {
                  return "Test262Error: " + this.message;
                };

                Test262Error.thrower = function (message) {
                  throw new Test262Error(message);
                };
            "#,
    );
}

#[test]
fn test_new() {
    let output = quick_run_script(
        r#"
                function MyConstructor(inner) {
                    this.inner = inner
                    return 'lol'
                }

                const obj = new MyConstructor(123)
                sink(obj.inner === 123)
                sink(obj.__proto__ === MyConstructor.prototype)
                sink(obj.constructor === MyConstructor)
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
        ],
    );
}

#[test]
fn test_temporal_dead_zone_none() {
    let _ = quick_run_script(
        r#"
                "use strict"
                let x = 12
                sink(foo())
                function foo() { return x }
            "#,
    );
}

#[test]
#[should_panic]
fn test_temporal_dead_zone() {
    let _ = quick_run_script(
        r#"
                "use strict"
                sink(foo())
                let x = 12
                function foo() { return x }
            "#,
    );
}

#[test]
fn test_reference_error() {
    let output = quick_run_script(
        r#"
                try {
                    aVariableThatDoesNotExist;
                } catch (e) {
                    sink(e instanceof ReferenceError);
                }
            "#,
    );

    assert_eq!(&output.sink, &[Some(Literal::Bool(true))]);
}

#[test]
fn test_unwinding_on_exception() {
    let output = quick_run_script(
        r#"
                try {
                    (function() {
                        throw 42;
                    })()
                } catch (e) {
                    sink(e);
                }
            "#,
    );

    assert_eq!(&output.sink, &[Some(Literal::Number(42.0))]);
}

#[test]
fn test_void_operator() {
    let output = quick_run_script("sink(123); sink(void 123);");
    assert_eq!(
        &output.sink,
        &[Some(Literal::Number(123.0)), Some(Literal::Undefined)]
    );
}

#[test]
fn test_eval_completion_value() {
    let output = quick_run_script("sink(eval('11'))");
    assert_eq!(&output.sink, &[Some(Literal::Number(11.0))]);
}

#[test]
fn test_array_properties() {
    let output = quick_run_script(
        r#"
            const arr = ['a', 123, false]
            for (const name in arr) {
              sink(name);
              sink(arr[name]);
            }
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::String("0".to_string())),
            Some(Literal::String("a".to_string())),
            Some(Literal::String("1".to_string())),
            Some(Literal::Number(123.0)),
            Some(Literal::String("2".to_string())),
            Some(Literal::Bool(false)),
        ]
    );
}

#[test]
fn test_generator_basic() {
    let output = quick_run_script(
        r#"
function* makeGenerator() {
  yield 'first';
  for (let i=0; i < 5; i++) {
    yield i * 2;
  }
  yield 123;
}

const generator = makeGenerator();
let item;
do {
  item = generator.next();
  sink(item.value);
} while (!item.done);
        "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::String("first".to_string())),
            Some(Literal::Number(0.0)),
            Some(Literal::Number(2.0)),
            Some(Literal::Number(4.0)),
            Some(Literal::Number(6.0)),
            Some(Literal::Number(8.0)),
            Some(Literal::Number(123.0)),
            Some(Literal::Undefined),
        ]
    );
}

#[test]
fn test_for_of_generator() {
    let output = quick_run_script(
        r#"
                function* getSomeNumbers() {
                    for (let i=0; i < 5; i++) {
                        yield i * 3;
                    }
                }

                for (const value of getSomeNumbers()) {
                    sink(value);
                }
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Number(0.0)),
            Some(Literal::Number(3.0)),
            Some(Literal::Number(6.0)),
            Some(Literal::Number(9.0)),
            Some(Literal::Number(12.0)),
        ]
    );
}

#[test]
fn test_short_circuiting_and() {
    let output = quick_run_script(
        r#"
                function getSomethingElse() { sink(2); return 456; }
                function getFalsy() { sink(1); return ""; }
                sink(getFalsy() && getSomethingElse());
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Number(1.0)),
            Some(Literal::String("".to_string())),
        ]
    );
}

#[test]
fn test_short_circuiting_or() {
    let output = quick_run_script(
        r#"
                function getSomethingElse() { sink(2); return 456; }
                function getTruthy() { sink(1); return 123; }
                sink(getTruthy() || getSomethingElse());
            "#,
    );

    assert_eq!(
        &output.sink,
        &[Some(Literal::Number(1.0)), Some(Literal::Number(123.0)),]
    );
}

#[test]
fn test_module_default_export() {
    let mut loader = loader::Loader::new_cwd();
    let root_fnid = loader
        .load_code_forced(
            loader::FileID::File(PathBuf::from("/virtualtest/root.mjs")),
            r#"
                    import the_thing from "./somewhere/the_thing.mjs";
                    sink(the_thing.the_content);
                "#
            .to_string(),
            bytecode_compiler::SourceType::Module,
        )
        .unwrap();

    loader
        .load_code_forced(
            loader::FileID::File(PathBuf::from("/virtualtest/somewhere/the_thing.mjs")),
            r#"
                    export default { the_content: 123 };
                "#
            .to_string(),
            bytecode_compiler::SourceType::Module,
        )
        .unwrap();

    let finished_data = complete_run(&mut loader, root_fnid);
    assert_eq!(&finished_data.sink, &[Some(Literal::Number(123.0)),]);
}

#[test]
fn test_module_named_export() {
    let mut loader = loader::Loader::new_cwd();
    let root_fnid = loader
        .load_code_forced(
            loader::FileID::File(PathBuf::from("/virtualtest/root.mjs")),
            r#"
                    import * as the_thing from "./somewhere/the_thing.mjs";
                    sink(the_thing.the_content);
                    sink(the_thing.double_the_content());
                "#
            .to_string(),
            bytecode_compiler::SourceType::Module,
        )
        .unwrap();

    loader
        .load_code_forced(
            loader::FileID::File(PathBuf::from("/virtualtest/somewhere/the_thing.mjs")),
            r#"
                    export const the_content = 123;
                    export function double_the_content() {
                        return 2 * the_content;
                    };
                "#
            .to_string(),
            bytecode_compiler::SourceType::Module,
        )
        .unwrap();

    let finished_data = complete_run(&mut loader, root_fnid);
    assert_eq!(
        &finished_data.sink,
        &[Some(Literal::Number(123.0)), Some(Literal::Number(246.0))]
    );
}

#[test]
fn test_delete_basic() {
    let output = quick_run_script(
        "
            const obj = {a: 123};
            sink(obj.a);
            delete obj.a;
            sink(obj.a);
        ",
    );
    assert_eq!(
        &output.sink,
        &[Some(Literal::Number(123.0)), Some(Literal::Undefined)]
    );
}

#[test]
fn test_to_number() {
    let output = quick_run_script(
        "
            sink(+123.0);
            sink(+{});
            sink(+null);
            sink(+undefined);
            sink(+true);
            sink(+false);
        ",
    );

    let expected = &[123.0, f64::NAN, 0.0, f64::NAN, 1.0, 0.0];

    assert_eq!(output.sink.len(), expected.len());
    for (out, &exp) in output.sink.iter().zip(expected) {
        let out = if let Some(Literal::Number(n)) = out {
            *n
        } else {
            panic!("expected Literal::Number, got {:?}", out);
        };

        // Check for equality, including NaN
        assert_eq!(f64::to_bits(out), f64::to_bits(exp));
    }
}

#[test]
fn test_autobox() {
    let output = quick_run_script(
        "
            function tryValue(value, expectedPrototype, sentinelValue) {
                expectedPrototype.sentinel = sentinelValue;
                sink(value.sentinel === sentinelValue);
            }
            tryValue(123.45,            Number.prototype,  'sentinel:number');
            tryValue(true,              Boolean.prototype, 'sentinel:boolean');
            tryValue('some string',     String.prototype,  'sentinel:string');
            // TODO fix this test when I implement 'Symbol'
            if (globalThis.Symbol) {
                throw new Error('re-enable this portion of the test!');
                // tryValue(Symbol.for('someSymbol'), Symbol.prototype, 'sentinel:symbol');
            }
            ",
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
        ]
    );
}

#[test]
fn test_autobox_failure() {
    quick_run_script(
            "
            function tryValue(value) {
                try {
                    Object.prototype.valueOf.call(value);
                    throw new Error('an exception should have been caught!');
                } catch (err) {
                    if (err.name !== 'TypeError') {
                        throw new Error('the exception should have been a TypeError, not a ' + err.name);
                    }
                }
            }
            tryValue(null);
            tryValue(undefined);
            ",
        );
}

#[test]
fn test_string_index() {
    let output = quick_run_script(
        "const s = 'asdlol123';
            sink(s[0]);
            sink(s[4]);
            sink(s[8]);
            sink(s[9]);",
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::String("a".to_string())),
            Some(Literal::String("o".to_string())),
            Some(Literal::String("3".to_string())),
            Some(Literal::Undefined),
        ]
    );
}

#[test]
#[allow(non_snake_case)]
fn test_string_codePointAt() {
    let output = quick_run_script(
        "const s = 'asdlol123';
            for (let i=0; i < s.length; ++i) {
                sink(s.codePointAt(i));
            }
            sink(s.codePointAt(s.length));
            ",
    );

    let ref_string = "asdlol123";
    let ref_string_u16: Vec<_> = ref_string.encode_utf16().collect();

    assert_eq!(output.sink.len(), ref_string.len() + 1);

    for (i, &code_point) in ref_string_u16.iter().enumerate() {
        assert_eq!(output.sink[i], Some(Literal::Number(code_point as f64)));
    }
    assert_eq!(output.sink[ref_string.len()], Some(Literal::Undefined));
}

#[test]
#[allow(non_snake_case)]
fn test_string_fromCodePoint() {
    let output = quick_run_script(
        "const s = 'asdlol123';
            for (let i=0; i < s.length; ++i) {
                sink(String.fromCodePoint(s.codePointAt(i)));
            }

            try {
                String.fromCodePoint(undefined);
            } catch (err) {
                sink(err.name)
            }
            ",
    );

    let ref_string = "asdlol123";

    assert_eq!(output.sink.len(), ref_string.len() + 1);

    for (i, ch) in ref_string.chars().enumerate() {
        assert_eq!(output.sink[i], Some(Literal::String(ch.into())));
    }
    assert_eq!(
        output.sink[ref_string.len()],
        Some(Literal::String("RangeError".into()))
    );
}

#[test]
#[allow(non_snake_case)]
fn test_RegExp_test() {
    let output = quick_run_script(
        r#"const re = /^\d{2,5}$/;
            sink(re.test('12'));
            sink(re.test('123'));
            sink(re.test('1234'));
            sink(re.test('12345'));
            sink(re.test('x12345'));
            sink(re.test('123456'));
            "#,
    );

    assert_eq!(
        &output.sink,
        &[
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
            Some(Literal::Bool(true)),
            Some(Literal::Bool(false)),
            Some(Literal::Bool(false)),
        ]
    );
}

#[test]
fn test_final_return_value() {
    let mut loader = loader::Loader::new_cwd();
    let mut realm = Realm::new(&mut loader);

    let init_script = r#"
            const myVariable = 123;
            globalThis.aFunction = function() { return myVariable; }
        "#;
    let chunk_fnid = loader
        .load_script_anon(init_script.to_string())
        .expect("couldn't compile test script");
    Interpreter::new(&mut realm, &mut loader, chunk_fnid)
        .run()
        .unwrap()
        .expect_finished();

    let global_obj = realm.global_obj;
    let inner_closure = realm
        .heap
        .get_chained(global_obj, heap::IndexOrKey::Key("aFunction"))
        .expect("no property `aFunction`")
        .value()
        .unwrap();
    let inner_closure = realm
        .heap
        .as_closure(inner_closure)
        .expect("`aFunction` not a closure");
    let inner_closure = Rc::clone(&inner_closure);

    let finish = Interpreter::new_call(
        &mut realm,
        &mut loader,
        inner_closure,
        Value::Undefined,
        &[],
    )
    .run()
    .unwrap()
    .expect_finished();

    let lit = try_value_to_literal(finish.ret_val, &realm.heap);
    assert_eq!(lit, Some(Literal::Number(123.0)));
}

#[test]
fn test_string_nonstring_concat() {
    let output = quick_run_script("sink('xx' + 99);");
    assert_eq!(&output.sink, &[Some(Literal::String("xx99".to_string()))]);
}

#[test]
fn test_nonstring_string_concat() {
    let output = quick_run_script(
        "
            const x = { valueOf: () => 'hello' };
            sink(99 + x)
        ",
    );
    assert_eq!(
        &output.sink,
        &[Some(Literal::String("99hello".to_string()))]
    );
}

#[test]
fn test_number_null_concat() {
    let output = quick_run_script("sink('1' + null); sink(null + '1');");
    assert_eq!(
        &output.sink,
        &[
            Some(Literal::String("1null".to_string())),
            Some(Literal::String("null1".to_string()))
        ]
    );
}

#[test]
fn test_number_nonnumber_add() {
    let output = quick_run_script(
        "
            const x = { valueOf: () => 99 };
            sink(99 + x)
        ",
    );
    assert_eq!(&output.sink, &[Some(Literal::Number(198.0))]);
}

#[test]
fn regression_cross_conversion_suspend() {
    // with the current impl, suspensions in JS code executed during a
    // native call (for example, primitive coercion) are broken: we don't get a
    // proper return value, and can't resume.

    let mut loader = loader::Loader::new_cwd();
    let text = "
            const x = { 
                valueOf() {
                    debugger;
                    return 99;
                }
            };
            sink(99 + x);
        "
    .to_string();
    let chunk_fnid = loader
        .load_script_anon(text)
        .expect("couldn't compile test script");

    let mut realm = Realm::new(&mut loader);
    let exit = Interpreter::new(&mut realm, &mut loader, chunk_fnid)
        .run()
        .unwrap();

    assert!(matches!(exit, Exit::Suspended { .. }));
}

mod debugging {
    use super::*;
    use crate::Loader;

    #[test]
    #[cfg(feature = "debugger")]
    fn test_inline_breakpoint() {
        const SOURCE_CODE: &str = r#"
                function foo() {
                    sink(1);
                    debugger;
                    sink(2);
                }

                foo();
            "#;

        let mut loader = Loader::new_cwd();
        let main_fnid = loader.load_script_anon(SOURCE_CODE.to_string()).unwrap();

        let mut realm = Realm::new(&mut loader);

        let exit = Interpreter::new(&mut realm, &mut loader, main_fnid)
            .run()
            .expect("interpreter failed");

        let intrp_state = match exit {
            Exit::Finished(_) => panic!("finished instead of interrupting"),
            Exit::Suspended { intrp_state, .. } => intrp_state,
        };

        assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);

        let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
            .run()
            .unwrap()
            .expect_finished();
        assert_eq!(
            &finish_data.sink,
            &[
                Some(bytecode::Literal::Number(1.0)),
                Some(bytecode::Literal::Number(2.0)),
            ]
        );
    }

    #[test]
    #[cfg(feature = "debugger")]
    fn test_pos_breakpoint() {
        let mut loader = Loader::new_cwd();

        const SOURCE_CODE: &str = r#"
                function foo() {
                    sink(1);
                    sink(2);
                }

                foo();
            "#;
        let main_fnid = loader.load_script_anon(SOURCE_CODE.to_string()).unwrap();

        // Hardcoded. Must be updated if breakme-0.js changes
        let pos = swc_common::BytePos(166);

        let mut realm = Realm::new(&mut loader);

        // Resolve into the (potentially multiple) GIIDs
        let break_range_ids: Vec<_> = loader
            .resolve_break_loc(main_fnid, pos)
            .unwrap()
            .into_iter()
            .map(|(brid, _)| brid)
            .collect();

        let mut dbg = debugger::DebuggingState::new();

        for brid in break_range_ids {
            dbg.set_source_breakpoint(brid, &loader).unwrap();

            let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);
            interpreter.set_debugging_state(&mut dbg);

            let intrp_state = match interpreter.run().expect("interpreter failed") {
                Exit::Finished(_) => panic!("interpreter finished instead of breaking"),
                Exit::Suspended { intrp_state, .. } => intrp_state,
            };

            let giid = debugger::giid(&intrp_state);
            eprintln!("we are at: {:?}; sink = {:?}", giid, intrp_state.sink);
            if giid.0 == bytecode::FnID(2) {
                assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);
            } else {
                assert_eq!(&intrp_state.sink, &[]);
            }

            let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
                .run()
                .expect("interpreter failed")
                .expect_finished();
            assert_eq!(
                &finish_data.sink,
                &[
                    Some(bytecode::Literal::Number(1.0)),
                    Some(bytecode::Literal::Number(2.0)),
                ]
            );
        }
    }
}
